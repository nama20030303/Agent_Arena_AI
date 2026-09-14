"""
Isolated execution for the Coding Lab (§20).

User code never runs inside the FastAPI worker. It runs in a *separate* short-lived
interpreter with:
  * its own temp directory as cwd (removed afterwards);
  * POSIX rlimits: address space, CPU seconds, file size, file descriptors;
  * a process group we can kill on timeout;
  * a scrubbed environment (no API keys, no tokens, no AWS/GCP credentials);
  * an in-process guard that disables sockets, subprocess and filesystem escapes;
  * static import screening before execution (banned modules per task).

That is a defence-in-depth approach for a personal, single-user tool. For a
multi-tenant deployment put this runner in a container/gVisor/Firecracker - the
interface (`run_python`) stays identical.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings

MARKER = "###MLEA_RESULT###"

BANNED_DEFAULT = ["socket", "subprocess", "shutil", "ctypes", "multiprocessing", "urllib", "http", "smtplib", "ftplib", "telnetlib"]
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)

RUNNER = r'''
import importlib.util
import io
import json
import sys
import time
import traceback
import unittest
from contextlib import redirect_stdout, redirect_stderr

MARKER = "__MLEA_MARKER__"
result = {"tests": [], "stdout": "", "stderr": "", "error": "", "import_error": "", "runtime_ms": 0}


def emit():
    sys.stdout.flush()
    print(MARKER)
    print(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load " + filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    start = time.time()
    sys.path.insert(0, ".")
    try:
        import guard  # noqa: F401  (patch sockets/subprocess before user code runs)
    except Exception as exc:
        result["stderr"] += "guard disabled: %s\n" % exc
    try:
        load("solution", "solution.py")
    except BaseException:
        result["import_error"] = traceback.format_exc(limit=6)
        result["runtime_ms"] = int((time.time() - start) * 1000)
        emit()
        return 1
    try:
        test_module = load("test_solution", "test_solution.py")
    except BaseException:
        result["error"] = "test loading failed:\n" + traceback.format_exc(limit=6)
        result["runtime_ms"] = int((time.time() - start) * 1000)
        emit()
        return 1

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    flat = []

    def walk(s):
        for item in s:
            if isinstance(item, unittest.TestSuite):
                walk(item)
            else:
                flat.append(item)

    walk(suite)
    out = io.StringIO()
    with redirect_stdout(out):
        for case in flat:
            t0 = time.time()
            record = {"name": str(case).split(" ", 1)[0], "passed": False, "message": ""}
            res = unittest.TestResult()
            try:
                case.run(res)
            except BaseException:
                record["message"] = traceback.format_exc(limit=4)
            record["passed"] = bool(res.wasSuccessful()) and not res.errors and not res.failures
            ms = int((time.time() - t0) * 1000)
            record["runtime_ms"] = ms
            if res.failures:
                record["message"] = str(res.failures[0][1]).strip().splitlines()[-1][:400]
            elif res.errors:
                record["message"] = str(res.errors[0][1]).strip().splitlines()[-1][:400]
            elif res.skipped:
                record["message"] = "skipped"
            result["tests"].append(record)
    result["stdout"] = out.getvalue()[-8000:]
    result["runtime_ms"] = int((time.time() - start) * 1000)
    emit()
    return 0 if all(t["passed"] for t in result["tests"]) and result["tests"] else 2


if __name__ == "__main__":
    sys.exit(main())
'''

GUARD = r'''
# Runtime guard: the learner's own modules may not touch sockets or spawn processes.
# Library code (e.g. pandas importing subprocess internally) is allowed, otherwise
# legitimate exercises would break. Defence in depth for a personal tool: rlimits,
# temp cwd, scrubbed env, static screening of user code, and this in-process guard.
import builtins
import sys

_BLOCKED = {
    'socket', 'ssl', 'subprocess', 'shutil', 'multiprocessing', 'telnetlib',
    'ftplib', 'smtplib', 'poplib', 'imaplib', 'ctypes',
}
_ROOTS = {'socket', 'ssl', 'subprocess', 'shutil', 'multiprocessing', 'telnetlib', 'ftplib', 'ctypes'}
_USER_MODULES = {'solution', 'test_solution', '__main__'}
_orig_import = builtins.__import__


def _module_of(frame, depth=8):
    for _ in range(depth):
        if frame is None:
            return ''
        name = frame.f_globals.get('__name__', '')
        if name:
            return name
        frame = frame.f_back
    return ''


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = (globals or {}).get('__name__') or _module_of(sys._getframe(1))
    if module in _USER_MODULES:
        root = (name or '').split('.')[0]
        if name in _BLOCKED or root in _ROOTS:
            raise ImportError('import %r is blocked in the sandbox' % name)
    return _orig_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import

try:
    import os as _os

    def _blocked(*args, **kwargs):
        raise PermissionError('process control is blocked in the sandbox')

    for _name in ('system', 'popen', 'execv', 'execve', 'execl', 'spawnl', 'fork'):
        if hasattr(_os, _name):
            setattr(_os, _name, _blocked)
except Exception:
    pass
'''


@dataclass
class TestOutcome:
    name: str
    passed: bool
    message: str = ""
    runtime_ms: int = 0


@dataclass
class RunResult:
    ok: bool
    total_tests: int = 0
    passed_tests: int = 0
    tests: list[TestOutcome] = field(default_factory=list)
    stdout: str = ""
    error: str = ""
    runtime_ms: int = 0
    exit_code: int = 0
    violated: list[str] = field(default_factory=list)
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tests"] = [asdict(t) for t in self.tests]
        return data


def screen_imports(code: str, *, extra_banned: list[str] | None = None) -> list[str]:
    banned = set(BANNED_DEFAULT) | {b.split(".")[0] for b in (extra_banned or [])}
    found: list[str] = []
    for match in IMPORT_RE.finditer(code or ""):
        module = match.group(1)
        root = module.split(".")[0]
        if module in banned or root in banned:
            found.append(module)
    return sorted(set(found))


def _preexec(memory_mb: int, cpu_seconds: int):  # pragma: no cover - platform dependent
    def configure() -> None:  # runs between fork and exec
        import resource

        def _limit(res: int, value: int) -> None:
            try:
                soft, hard = resource.getrlimit(res)
                resource.setrlimit(res, (min(value, hard if hard != resource.RLIM_INFINITY else value), hard))
            except (ValueError, OSError):
                pass

        if memory_mb > 0:
            _limit(resource.RLIMIT_AS, memory_mb * 1024 * 1024)
        if cpu_seconds > 0:
            _limit(resource.RLIMIT_CPU, cpu_seconds)
        _limit(resource.RLIMIT_FSIZE, 8 * 1024 * 1024)
        _limit(resource.RLIMIT_NOFILE, 64)
        try:
            os.setsid()
        except OSError:
            pass

    return configure


def run_python(
    solution_code: str,
    tests_code: str,
    *,
    timeout: int | None = None,
    memory_mb: int | None = None,
    banned_imports: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
    data_files: dict[str, str] | None = None,
) -> RunResult:
    """Execute solution + tests in a throwaway interpreter."""
    timeout = timeout or settings.code_execution_timeout_seconds
    memory_mb = memory_mb or settings.code_execution_memory_mb
    if not settings.code_execution_enabled:
        return RunResult(ok=False, error="Code execution is disabled on this server (CODE_EXECUTION_ENABLED=false).")

    violations = screen_imports(solution_code, extra_banned=banned_imports) + screen_imports(tests_code)
    violations = sorted(set(violations))
    if violations:
        return RunResult(
            ok=False,
            error="Blocked imports detected: " + ", ".join(violations) + ". Remove them and try again.",
            violated=violations,
        )

    workdir = Path(tempfile.mkdtemp(prefix="mlea_", dir=str(settings.sandbox_dir)))
    try:
        (workdir / "solution.py").write_text(solution_code or "", encoding="utf-8")
        (workdir / "test_solution.py").write_text(tests_code or "", encoding="utf-8")
        (workdir / "guard.py").write_text(GUARD, encoding="utf-8")
        (workdir / "runner.py").write_text(RUNNER.replace("__MLEA_MARKER__", MARKER), encoding="utf-8")
        for name, content in (extra_files or {}).items():
            target = (workdir / name).resolve()
            if workdir.resolve() not in target.parents and target != workdir.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for name, content in (data_files or {}).items():
            (workdir / name).write_text(content, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "LANG": "C.UTF-8",
            "MLA_SANDBOX": "1",
            "PYTHONPATH": str(workdir),
        }
        # inherit site-packages location but nothing secret
        for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            value = os.environ.get(key)
            if value:
                env[key] = value

        started = time.perf_counter()
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-E", "-s", "-B", "runner.py"],
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_preexec(memory_mb, max(1, int(timeout))) if hasattr(os, "setsid") else None,
                start_new_session=True,
            )
        except OSError as exc:
            return RunResult(ok=False, error=f"could not start sandbox: {exc}")

        try:
            stdout, stderr = process.communicate(timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()
            stdout, stderr = "", ""
            try:
                process.communicate(timeout=3)
            except Exception:  # noqa: BLE001
                pass
            return RunResult(
                ok=False,
                timed_out=True,
                error=f"Execution exceeded the {timeout}s limit (killed). "
                "Avoid unbounded loops; vectorise instead of iterating in Python.",
                runtime_ms=int((time.perf_counter() - started) * 1000),
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if process.returncode is not None and process.returncode < 0 and MARKER not in (stdout or ""):
            import signal as _signal

            try:
                name = _signal.Signals(-int(process.returncode)).name
            except (ValueError, AttributeError):
                name = f"signal {-int(process.returncode)}"
            limited = name in {"SIGKILL", "SIGXCPU", "SIGXFSZ", "SIGSEGV", "SIGBUS"}
            return RunResult(
                ok=False,
                timed_out=name in {"SIGKILL", "SIGXCPU"},
                error=(
                    f"The sandbox terminated the run with {name}. This usually means the CPU or memory "
                    f"limit was hit ({settings.code_execution_timeout_seconds}s / {settings.code_execution_memory_mb}MB): "
                    "check for an unbounded loop or a huge temporary array, and vectorise instead of iterating."
                )
                + ((chr(10) + (stderr or "")[-1500:]) if stderr else ""),
                exit_code=int(process.returncode),
                runtime_ms=elapsed_ms,
                stdout=(stdout or "")[: settings.code_execution_max_output_chars],
            )
        payload: dict[str, Any] | None = None
        if MARKER in stdout:
            blob = stdout.split(MARKER, 1)[1].strip()
            try:
                import json as _json

                payload = _json.loads(blob.splitlines()[0] if blob else "{}")
            except Exception:  # noqa: BLE001
                payload = None
        if payload is None:
            message = (stderr or "")[-3000:] or (stdout or "")[-2000:]
            truncated = "Memory limit" in message or "MemoryError" in message or "Killed" in message
            return RunResult(
                ok=False,
                error=(
                    "The sandbox killed the run (usually the memory limit was exceeded)."
                    if truncated
                    else f"Sandbox did not return results (exit {process.returncode}).\n{message[:1500] or 'no output'}"
                ),
                exit_code=int(process.returncode or 0),
                runtime_ms=elapsed_ms,
                stdout=(stdout or "")[: settings.code_execution_max_output_chars],
            )

        tests = [
            TestOutcome(
                name=str(t.get("name") or f"test_{i + 1}"),
                passed=bool(t.get("passed")),
                message=str(t.get("message") or ""),
                runtime_ms=int(t.get("runtime_ms") or 0),
            )
            for i, t in enumerate(payload.get("tests") or [])
        ]
        import_error = str(payload.get("import_error") or "")
        error = str(payload.get("error") or "")
        if import_error:
            error = "Your solution raised while importing:\n" + import_error[:2500]
        total = len(tests)
        passed = sum(1 for t in tests if t.passed)
        ok = bool(total) and passed == total and not error
        return RunResult(
            ok=ok,
            total_tests=total,
            passed_tests=passed,
            tests=tests,
            stdout=str(payload.get("stdout") or "")[: settings.code_execution_max_output_chars],
            error=error,
            runtime_ms=elapsed_ms,
            exit_code=int(process.returncode or 0),
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def sandbox_capabilities() -> dict[str, Any]:
    """Report what the sandbox can do - the UI shows this instead of guessing."""
    try:
        probe = run_python(
            "import numpy, sys\nVALUE = numpy.arange(6).reshape(2,3).mean()\nSYS = sys.version_info[0]",
            "import unittest\nfrom solution import VALUE, SYS\n"
            "class P(unittest.TestCase):\n    def test_numpy(self):\n        self.assertAlmostEqual(VALUE, 2.5)\n"
            "    def test_py(self):\n        self.assertGreaterEqual(SYS, 3)\n",
            timeout=20,
        )
        numpy_ok = probe.ok
    except Exception:  # noqa: BLE001
        numpy_ok = False
    try:
        pandas_probe = run_python("import pandas\nROWS=len(pandas.DataFrame({'a':[1,2]}))", "import unittest\nfrom solution import ROWS\nclass P(unittest.TestCase):\n    def test_rows(self):\n        self.assertEqual(ROWS, 2)\n", timeout=25)
        pandas_ok = pandas_probe.ok
    except Exception:  # noqa: BLE001
        pandas_ok = False
    try:
        sklearn_probe = run_python("import sklearn\nV=1", "import unittest\nfrom solution import V\nclass P(unittest.TestCase):\n    def test_v(self):\n        self.assertEqual(V,1)\n", timeout=25)
        sklearn_ok = sklearn_probe.ok
    except Exception:  # noqa: BLE001
        sklearn_ok = False
    return {
        "enabled": settings.code_execution_enabled,
        "isolation": "subprocess + rlimits + import guard + temp dir + scrubbed env",
        "timeout_seconds": settings.code_execution_timeout_seconds,
        "memory_mb": settings.code_execution_memory_mb,
        "numpy": numpy_ok,
        "pandas": pandas_ok,
        "sklearn": sklearn_ok,
        "language": "python",
        "posix_limits": hasattr(os, "setsid"),
    }
