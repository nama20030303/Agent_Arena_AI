"""
Offline ("no AI") engine.

This is what the product does when there is no LLM, no budget or no internet. It is
*not* a stub: it produces real explanations, real questions and real grading by
working directly over the indexed library with deterministic NLP. Requirement §47
("fallback without AI") is tested against these functions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

from app.knowledge.retrieval import RetrievedChunk
from app.knowledge.textutil import normalize, split_sentences, tokenize

DEF_RE = re.compile(
    r"^(?P<term>[A-ZА-ЯЁ][\w \-/'()]{2,48}?)\s+(?:is|are|means|refers to|describes|определяется как|это|является)\s+(?P<body>.{20,300})$",
)
FORMULA_RE = re.compile(r"(\b(w|y|L|J|E|theta|theta|x|a|b)\s*=|\\nabla|∇|sum_|arg\s?min|arg\s?max|\bO\()")
STEP_RE = re.compile(r"^(?:\d+[.)]|[-*•])\s+(.{10,220})$")
CODE_FENCE_RE = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
TERM_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "we", "you", "it", "is", "are", "of", "and", "or",
    "in", "on", "for", "with", "by", "to", "from", "as", "at", "be", "can", "may", "model", "data",
}


def _title_term(sentence: str) -> str:
    m = DEF_RE.match(sentence.strip())
    if m:
        return m.group("term").strip()
    words = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё][\w\-']{2,}", sentence) if w.lower() not in TERM_STOP]
    return " ".join(words[:3]) if words else ""


# --------------------------------------------------------------------------- #
#  Explanation
# --------------------------------------------------------------------------- #
def local_explanation(
    *,
    topic: str,
    level: int,
    results: list[RetrievedChunk],
    style: str = "explain",
    ai_hint: str = "",
) -> dict[str, Any]:
    """Compose a grounded explanation from retrieved chunks, with citations."""
    topic_terms = set(tokenize(topic))
    topic_low = topic.lower()

    definitions: list[tuple[RetrievedChunk, str, str]] = []
    formulas: list[tuple[RetrievedChunk, str]] = []
    steps: list[str] = []
    code_blocks: list[tuple[RetrievedChunk, str]] = []
    generic: list[tuple[RetrievedChunk, str]] = []

    for chunk in results:
        for sentence in split_sentences(chunk.text):
            low = sentence.lower()
            related = bool(topic_terms & set(tokenize(sentence))) or topic_low in low or not topic_low
            if not related:
                continue
            if len(sentence) > 420:
                sentence = sentence[:417] + "..."
            if DEF_RE.match(sentence.strip()):
                definitions.append((chunk, sentence, _title_term(sentence)))
            elif FORMULA_RE.search(sentence) and len(sentence) < 240:
                formulas.append((chunk, sentence))
            elif STEP_RE.match(sentence.strip()):
                steps.extend(m.group(1) for m in (STEP_RE.match(s.strip()) for s in [sentence]) if m)
            elif len(sentence) > 60:
                generic.append((chunk, sentence))
        for block in CODE_FENCE_RE.findall(chunk.text):
            if len(block.strip()) > 40:
                code_blocks.append((chunk, block.strip()[:1200]))

    if not definitions and not generic and not formulas:
        return {
            "text": _no_material_message(topic, results),
            "sources": [],
            "engine": "local",
            "grounded": False,
        }

    used_chunks: list[RetrievedChunk] = []
    def cite(chunk: RetrievedChunk) -> int:
        if chunk in used_chunks:
            return used_chunks.index(chunk) + 1
        used_chunks.append(chunk)
        return len(used_chunks)

    lines: list[str] = [f"### {topic}", ""]
    if level <= 1:
        lines.append("*Offline explanation assembled from your library, levelled to a beginner (no AI used).*")
    else:
        lines.append(f"*Offline explanation assembled from your library (level {level}/10, no AI used).*")
    lines.append("")

    section_title = {
        "simple": "In plain words",
        "deep_dive": "What your sources actually say",
        "code": "Working definition",
        "interview": "Short answer",
    }.get(style, "What it is")

    lines.append(f"**{section_title}**")
    picked: list[str] = []
    seen_norm: set[str] = set()
    for chunk, sentence, _term in definitions[:4] if level <= 3 else definitions[:6]:
        key = normalize(sentence)[:70]
        if key in seen_norm:
            continue
        seen_norm.add(key)
        idx = cite(chunk)
        picked.append(f"- {sentence} [{idx}]")
    if not picked:
        for chunk, sentence in generic[:4]:
            key = normalize(sentence)[:70]
            if key in seen_norm:
                continue
            seen_norm.add(key)
            idx = cite(chunk)
            picked.append(f"- {sentence} [{idx}]")
    lines.extend(picked or ["- (nothing definitional found in the retrieved passages - see excerpts below)"])
    lines.append("")

    if style in {"explain", "deep_dive", "code"} and formulas:
        lines.append("**Formalism your sources use**")
        for chunk, sentence in formulas[:4]:
            lines.append(f"- `{normalize(sentence)[:200]}` [{cite(chunk)}]")
        lines.append("")

    ordered_steps = [s for s in steps if s][:6]
    if ordered_steps and style != "simple":
        lines.append("**The procedure, as written in the material**")
        for i, step in enumerate(ordered_steps, start=1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if code_blocks and style in {"code", "explain", "deep_dive"}:
        chunk, block = code_blocks[0]
        lines.append(f"**Code from your material** [{cite(chunk)}]")
        lines.append("```python")
        lines.append(block)
        lines.append("```")
        lines.append("")

    if level <= 2:
        lines.append("**Analogy**")
        lines.append(
            "- Think of it as a loop that keeps a quantity under control: you look at the current error, "
            "move a little in the direction that reduces it, and check again. "
            "*This framing is mine, not a quotation from your library.*"
        )
        lines.append("")

    if style == "deep_dive" and generic:
        lines.append("**Excerpts worth reading in full**")
        for chunk, sentence in generic[3:7]:
            lines.append(f"- {sentence[:260]} [{cite(chunk)}]")
        lines.append("")

    lines.append("**Check your understanding**")
    for i, q in enumerate(local_questions_from_chunks(topic=topic, results=results, count=3), start=1):
        lines.append(f"{i}. {q['stem']}")
    lines.append("")
    lines.append(
        "> Answer these in the Teacher chat (mode *Review my answer*) and the Learning Engine will update "
        "your skill scores and schedule the next review."
    )
    if ai_hint:
        lines.append("")
        lines.append(ai_hint)

    return {
        "text": "\n".join(lines).strip(),
        "sources": [c.citation for c in used_chunks],
        "engine": "local",
        "grounded": True,
        "sections": ["definition", "formalism", "procedure", "check"],
    }


def _no_material_message(topic: str, results: list[RetrievedChunk]) -> str:
    extra = (
        "\n\nRelated but weaker hits (so you can judge for yourself):\n"
        + "\n".join(f"- {r.document_title} · {r.chapter or r.section}: {r.text[:160]}…" for r in results[:3])
        if results
        else ""
    )
    return (
        f"### {topic}\n\n"
        "Your library does not contain material on this topic, so I will not pretend that it does.\n\n"
        "What you can do:\n"
        "1. Upload a book/PDF/EPUB or add a documentation URL that covers it (Library → Upload / Add URL).\n"
        "2. Ask me to use the *web sources* flow if you allowed web retrieval.\n"
        "3. Configure an AI provider in `.env` to get a general explanation (which I would then clearly mark as "
        "not coming from your sources)."
        + extra
    )


# --------------------------------------------------------------------------- #
#  Question generation
# --------------------------------------------------------------------------- #
def local_questions_from_chunks(*, topic: str, results: list[RetrievedChunk], count: int = 4, difficulty: int = 2) -> list[dict[str, Any]]:
    """
    Template question generation over retrieved text: cloze, definitional, numeric,
    contrast. Every generated question keeps a citation so it can be traced back.
    """
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    topic_low = (topic or "").lower()
    topic_terms = [t for t in tokenize(topic) if t not in TERM_STOP]

    for chunk in results:
        for sentence in split_sentences(chunk.text):
            if len(questions) >= count * 3:
                break
            norm = normalize(sentence).lower()
            if norm in seen or len(sentence) < 70 or len(sentence) > 330:
                continue
            # 1) cloze deletion on the most specific term
            terms = [t for t in set(tokenize(sentence)) if len(t) >= 7 and t not in TERM_STOP]
            anchor = _best_cloze_term(sentence, topic_terms, terms)
            if anchor and difficulty <= 3:
                masked = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(anchor)}(?![A-Za-z0-9_])", "____", sentence, count=1, flags=re.IGNORECASE)
                if masked != sentence:
                    seen.add(norm)
                    questions.append(
                        {
                            "type": "conceptual",
                            "difficulty": max(1, difficulty),
                            "stem": f"Fill the gap: “{masked}”",
                            "expected_answer": anchor,
                            "expected_value": anchor.lower(),
                            "tolerance": 0.0,
                            "expected_points": [anchor.lower()],
                            "explanation": f"The passage in {chunk.document_title}{', ' + chunk.chapter if chunk.chapter else ''} uses “{anchor}” here.",
                            "skills": [topic] if topic else [],
                            "citation": chunk.citation,
                            "generated_by": "template",
                        }
                    )
                    continue
            # 2) numeric reasoning
            numbers = NUMBER_RE.findall(sentence)
            if len(numbers) >= 2 and difficulty >= 2:
                seen.add(norm)
                questions.append(
                    {
                        "type": "math",
                        "difficulty": max(2, difficulty),
                        "stem": "Compute from this passage: "
                        + sentence[:280]
                        + "\n\nQuestion: what are the numeric quantities stated, and how are they related? "
                        "Give the value you derive and the one-line reasoning.",
                        "expected_answer": "Reproduce the arithmetic implied by the passage; any value consistent with the stated numbers is accepted.",
                        "expected_points": [n for n in numbers[:4]],
                        "explanation": f"Numbers appear in {chunk.document_title}"
                        + (f" · {chunk.chapter}" if chunk.chapter else ""),
                        "skills": [topic] if topic else [],
                        "citation": chunk.citation,
                        "generated_by": "template",
                    }
                )
                continue
            # 3) "why/what for" from a definitional sentence
            m = DEF_RE.match(sentence.strip())
            if m and topic_low in norm or (m and topic_terms and set(topic_terms) & set(tokenize(m.group("term").lower()))):
                seen.add(norm)
                term = m.group("term").strip()
                questions.append(
                    {
                        "type": "open",
                        "difficulty": max(2, difficulty),
                        "stem": f"Explain in your own words: what is {term}, and why does it matter here? "
                        "(Do not look at your notes. Then compare with your sources at the end.)",
                        "expected_answer": sentence[:300],
                        "expected_points": [p for p in tokenize(m.group("body"))[:8]],
                        "explanation": "Source text: " + sentence[:280],
                        "skills": [topic] if topic else [],
                        "citation": chunk.citation,
                        "generated_by": "template",
                    }
                )
    # 4) contrast question across two documents (also surfaces conflicts, §31)
    if len({c.document_id for c in results}) >= 2 and len(questions) < count:
        a, b = results[0], next((c for c in results[1:] if c.document_id != a.document_id), None)
        if b is not None:
            questions.append(
                {
                    "type": "conceptual",
                    "difficulty": 3,
                    "stem": f"Compare how these two sources frame {topic or 'the topic'}:\n"
                    f"- {a.document_title}: {a.text[:180]}…\n"
                    f"- {b.document_title}: {b.text[:180]}…\n"
                    "Where do they agree, where do they emphasise different failure modes, and which one is closer "
                    "to production reality?",
                    "expected_answer": "A correct answer names the shared core, the different emphasis, and picks one with a reason.",
                    "expected_points": ["shared core", "different emphasis", "reasoned choice"],
                    "explanation": "Cross-source comparison built from your two documents.",
                    "skills": [topic] if topic else [],
                    "citation": a.citation,
                    "generated_by": "template",
                }
            )
    return questions[: max(1, count)]


def _best_cloze_term(sentence: str, topic_terms: list[str], candidates: list[str]) -> str:
    if not candidates:
        return ""
    scored = []
    for cand in candidates:
        score = len(cand)
        if cand.lower() in {t.lower() for t in topic_terms}:
            score += 8
        if cand in sentence and sentence.count(cand) == 1:
            score += 4
        if re.match(r"^[a-z_]+\.[a-z_]+\(?", cand):
            score += 5
        if cand.lower() in {"the", "this", "that", "with", "from", "using", "used", "models", "value"}:
            score -= 10
        scored.append((score, cand))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 6 else ""


# --------------------------------------------------------------------------- #
#  Evaluation (grading without an LLM)
# --------------------------------------------------------------------------- #
def evaluate_locally(*, question: dict[str, Any], answer: str, time_seconds: float = 0.0) -> dict[str, Any]:
    """Deterministic grading with the same output schema the AI evaluator returns."""
    answer_raw = (answer or "").strip()
    qtype = (question.get("type") or question.get("question_type") or "open").lower()
    points = [p for p in (question.get("expected_points") or []) if p]
    expected_value = str(question.get("expected_value") or "").strip()
    expected_answer = str(question.get("expected_answer") or "").strip()
    tolerance = float(question.get("tolerance") or 0.0)

    dims = {"correctness": 0.0, "depth": 0.0, "reasoning": 0.0, "precision": 0.0, "confidence": 0.0}
    error_type = "none"
    what_wrong = ""
    correction = ""

    if not answer_raw:
        dims.update({"correctness": 0, "depth": 0, "reasoning": 0, "precision": 0, "confidence": 0})
        return _verdict(
            correct=False, score=0, dims=dims, error_type="incomplete",
            what_is_wrong="No answer given.", correction="Attempt even a partial answer - the engine learns from your reasoning.",
            feedback="Empty answer. Start from what you do know and state your assumptions.",
            missing_points=points[:4], question=question, time_seconds=time_seconds,
        )

    if qtype == "mcq" and question.get("options"):
        correct_option = question.get("correct_option")
        choice = _parse_choice(answer_raw)
        ok = choice is not None and correct_option is not None and int(choice) == int(correct_option)
        dims["correctness"] = 100 if ok else 0
        dims["precision"] = 100 if ok else 20
        dims["depth"] = 35 if len(answer_raw) > 60 else 10
        dims["reasoning"] = min(100, 20 + len(tokenize(answer_raw)) // 2)
        dims["confidence"] = 70 if ok else 45
        if not ok:
            error_type = "conceptual"
            what_wrong = f"Selected option {choice if choice is not None else 'n/a'}; the keyed answer is option {correct_option}."
            correction = _option_text(question, correct_option)
        return _verdict(correct=ok, score=round(0.75 * dims["correctness"] + 0.15 * dims["depth"] + 0.1 * dims["precision"]),
                        dims=dims, error_type=error_type, what_is_wrong=what_wrong, correction=correction,
                        feedback=("Correct. " if ok else "Not correct. ") + (question.get("explanation") or "")[:400],
                        missing_points=[], question=question, time_seconds=time_seconds)

    if expected_value and _is_numeric(expected_value):
        given = _last_number(answer_raw)
        target = float(expected_value)
        ok = given is not None and abs(given - target) <= max(tolerance, abs(target) * 0.02, 1e-6)
        dims["correctness"] = 100 if ok else 0
        dims["precision"] = 100 if ok else (60 if given is not None else 0)
        dims["reasoning"] = min(100, 30 + 2 * len(tokenize(answer_raw)) // 3)
        dims["depth"] = 40 if len(answer_raw) > 120 else 15
        dims["confidence"] = 80 if ok else 40
        if not ok:
            error_type = "math"
            what_wrong = "Your final number does not match the expected value." if given is not None else "No final numeric value found in your answer."
            correction = f"Expected {expected_value}" + (f" ± {tolerance}" if tolerance else "") + ". Recompute the last step and show the value explicitly."
        return _verdict(correct=ok, score=round(0.7 * dims["correctness"] + 0.2 * dims["precision"] + 0.1 * dims["reasoning"]),
                        dims=dims, error_type=error_type, what_is_wrong=what_wrong, correction=correction,
                        feedback=f"Numeric check against the keyed value ({expected_value}).",
                        missing_points=[], question=question, time_seconds=time_seconds)

    # free text: coverage of expected points + answer-answer similarity
    answer_terms = set(tokenize(answer_raw))
    covered, missing = _coverage(answer_terms, points)
    similarity = 0.0
    if expected_answer:
        exp_terms = set(tokenize(expected_answer))
        if exp_terms:
            similarity = len(answer_terms & exp_terms) / math.sqrt(max(1, len(answer_terms)) * max(1, len(exp_terms)))
    ratio = covered / max(1, len(points)) if points else min(1.0, similarity * 1.8)
    word_count = len(answer_raw.split())

    dims["correctness"] = round(min(100.0, 100 * (0.55 * ratio + 0.45 * min(1.0, similarity * 2.0))), 1)
    dims["depth"] = round(min(100.0, 22 + 3.0 * covered + min(40, word_count / 3.2)), 1)
    dims["reasoning"] = round(min(100.0, 18 + 2.2 * _reasoning_markers(answer_raw) + min(35, word_count / 4.0)), 1)
    dims["precision"] = round(min(100.0, 20 + 34 * ratio + 18 * min(1.0, similarity * 2.2) + (10 if _has_number(answer_raw) else 0)), 1)
    dims["confidence"] = round(min(100.0, 30 + 4.5 * covered + (18 if "not sure" not in answer_raw.lower() else -10) + min(25, word_count / 6)), 1)

    score = round(0.42 * dims["correctness"] + 0.18 * dims["depth"] + 0.18 * dims["reasoning"] + 0.22 * dims["precision"] + 0.0 * dims["confidence"], 1)
    correct = score >= 62 and ratio >= (0.5 if points else 0.34)

    if not correct:
        if points and missing:
            error_type = "incomplete" if score >= 45 else "conceptual"
            what_wrong = "Missing: " + "; ".join(m[:120] for m in missing[:3])
            correction = f"To answer this you need to add {len(missing)} idea(s): " + "; ".join(m[:90] for m in missing[:3]) + "."
        elif word_count < 12:
            error_type = "incomplete"
            what_wrong = "Answer is too short to demonstrate the reasoning the question asks for."
            correction = "State the mechanism, then one concrete consequence of getting it wrong."
        else:
            error_type = "terminology"
            what_wrong = "Your wording does not line up with the terms the sources use."
            correction = "Re-answer using the vocabulary from the cited passage; precision is graded, not just vibes."
    else:
        if score < 80:
            what_wrong = "Mostly right but shallow: the missing pieces above are what an interviewer would push on."
            correction = ""

    feedback = (
        ("Good: the answer covers the key points" if correct else "Not there yet")
        + f" (coverage {round(100 * ratio)}%, {word_count} words, {round(dims['precision'])} precision). "
        + (f"Model answer: {expected_answer[:260]}" if not correct and expected_answer else "")
    )
    return _verdict(correct=correct, score=score, dims=dims, error_type=error_type, what_is_wrong=what_wrong,
                    correction=correction, feedback=feedback, missing_points=missing[:5], question=question,
                    time_seconds=time_seconds)


def _verdict(*, correct: bool, score: float, dims: dict[str, float], error_type: str, what_is_wrong: str,
             correction: str, feedback: str, missing_points: list[str], question: dict[str, Any], time_seconds: float) -> dict[str, Any]:
    check = similar_question(question, error_type)
    return {
        "correct": bool(correct),
        "score": float(round(score, 1)),
        "dimensions": {k: float(v) for k, v in dims.items()},
        "error_type": error_type,
        "what_is_wrong": what_is_wrong,
        "correction": correction,
        "feedback": feedback,
        "missing_points": missing_points,
        "check_question": check,
        "evaluated_by": "local",
        "time_seconds": time_seconds,
    }


def similar_question(question: dict[str, Any], error_type: str) -> dict[str, Any] | None:
    """§17: after a wrong answer, re-test with a similar question."""
    stem = (question.get("stem") or "").strip()
    if not stem:
        return None
    topic = question.get("topic") or (question.get("skills") or ["the same concept"])[0]
    variants = {
        "math": f"Redo the same computation, but write every intermediate value out. Topic: {topic}. "
                "Which step changes the result most, and why?",
        "conceptual": f"In one paragraph, explain {topic} to someone who has never heard of it, then give one case "
                      "where the naive intuition fails.",
        "incomplete": f"Re-answer: “{stem[:220]}” — this time list the required points explicitly before you explain them.",
        "terminology": f"Define the terms involved in “{stem[:160]}” first, then answer using exactly those terms.",
        "code_bug": "State the invariant the code must maintain, then walk the failing input through it line by line.",
        "wrong_assumption": f"Which assumption in your answer about {topic} would a production system break first? Justify.",
        "overconfident": f"Why might your answer about {topic} be wrong? Give one counter-example.",
    }
    return {
        "stem": variants.get(error_type, variants["incomplete"]),
        "type": question.get("type") or "open",
        "expected_answer": question.get("expected_answer") or "",
        "expected_points": question.get("expected_points") or [],
        "options": question.get("options") or [],
        "correct_option": question.get("correct_option"),
        "difficulty": max(1, int(question.get("difficulty") or 2)),
        "skills": question.get("skills") or [topic],
        "citation": question.get("citation") or {},
        "retry_of": question.get("id"),
    }


# --------------------------------------------------------------------------- #
#  Small text utils
# --------------------------------------------------------------------------- #
def _coverage(answer_terms: set[str], points: Iterable[str]) -> tuple[int, list[str]]:
    covered = 0
    missing: list[str] = []
    for point in points:
        pterms = {t for t in tokenize(str(point)) if len(t) > 3}
        if not pterms:
            covered += 1
            continue
        hits = len(pterms & answer_terms) / len(pterms)
        if hits >= 0.34:
            covered += 1
        else:
            missing.append(str(point))
    return covered, missing


def _reasoning_markers(text: str) -> int:
    markers = ("because", "since", "therefore", "thus", "so that", "which means", "if ", "then ", "however",
               "because of", "причина", "поэтому", "следовательно", "из-за", "из-за чего", "из-за")
    low = text.lower()
    return sum(low.count(m) for m in markers)


def _has_number(text: str) -> bool:
    return bool(NUMBER_RE.search(text))


def _parse_choice(text: str) -> int | None:
    stripped = text.strip().lower()
    m = re.match(r"^[\(\[]?([0-9]{1,2}|[a-d])[)\].:]?", stripped)
    if not m:
        m = re.search(r"option\s*([0-9]{1,2}|[a-d])", stripped)
    if not m:
        return None
    token = m.group(1)
    if token.isdigit():
        return int(token)
    return ord(token) - ord("a")


def _option_text(question: dict[str, Any], index: Any) -> str:
    options = question.get("options") or []
    if index is None or not options:
        return ""
    try:
        i = int(index)
    except (TypeError, ValueError):
        return ""
    if 0 <= i < len(options):
        opt = options[i]
        return opt.get("text", str(opt)) if isinstance(opt, dict) else str(opt)
    return ""


def _is_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def _last_number(text: str) -> float | None:
    matches = NUMBER_RE.findall(text.replace("**", ""))
    for raw in reversed(matches):
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            continue
    return None


def local_summary(text: str, *, max_bullets: int = 8) -> dict[str, Any]:
    """Extractive summary with the same shape as AI summaries (spec §7 summarize)."""
    sentences = split_sentences(normalize(text))
    if not sentences:
        return {"text": "", "bullets": [], "definitions": [], "questions": []}
    freq: Counter[str] = Counter()
    for s in sentences:
        freq.update(t for t in set(tokenize(s)) if t not in TERM_STOP)
    scored = []
    for i, s in enumerate(sentences):
        terms = [t for t in set(tokenize(s)) if t not in TERM_STOP]
        if not terms:
            continue
        score = sum(freq[t] for t in terms) / (len(terms) ** 0.6)
        score *= 1.0 + (0.12 if DEF_RE.match(s.strip()) else 0.0)
        score *= 1.0 + (0.1 if len(s) > 90 else -0.25)
        scored.append((score, i, s))
    scored.sort(reverse=True)
    chosen = [s for _score, _i, s in scored[:max_bullets]]
    chosen_order = [sentences.index(c) for c in chosen if c in sentences]
    chosen = [sentences[i] for i in sorted(set(chosen_order))]
    definitions = [s for s in chosen if DEF_RE.match(s.strip())][:4]
    questions = [
        f"Explain, without notes: what role does “{_title_term(s) or 'this idea'}” play in this material?"
        for s in chosen[:3]
    ]
    body = "\n".join(f"- {s}" for s in chosen)
    return {
        "text": body,
        "bullets": chosen,
        "definitions": definitions,
        "questions": questions,
        "engine": "local",
    }


def local_practice(topic: str, level: str, results: list[RetrievedChunk]) -> dict[str, Any] | None:
    """Derive a practice task from the material when no AI is available."""
    if not results:
        return None
    anchor = results[0]
    numbers = NUMBER_RE.findall(anchor.text)
    level_idx = {"beginner": 1, "intermediate": 2, "advanced": 3, "production": 4}.get(level, 2)
    if len(numbers) >= 2 and level_idx <= 2:
        return {
            "title": f"{topic}: hand-compute from the passage",
            "statement": f"Read this excerpt and reproduce the numbers it assumes:\n\n> {anchor.text[:600]}\n\n"
            "Task: identify the quantities, write the formula that links them, and compute the result step by step.",
            "expected_answer": "Any consistent derivation using the quantities in the excerpt.",
            "steps_required": ["list quantities with units", "state the formula", "substitute", "compute", "sanity-check the magnitude"],
            "starter_code": "import numpy as np\n\n# re-implement the computation from the passage\n",
            "hints": ["Start by writing down every symbol you can see in the excerpt.", "Check units before computing."],
            "difficulty": max(1, min(5, 1 + level_idx)),
            "est_minutes": 15,
            "kind": "math",
            "citation": anchor.citation,
        }
    return {
        "title": f"{topic}: implement from your sources",
        "statement": f"Using the algorithm description in your library:\n\n> {anchor.text[:700]}\n\n"
        "Implement it in Python without calling a library that does it for you. Then answer: which line makes the "
        "difference between the naive version and a numerically sane version?",
        "expected_answer": "A working implementation plus an explicit note on the stability/edge-case line.",
        "steps_required": ["parse the input format", "implement the core update rule", "handle empty/edge case", "verify on a tiny hand example"],
        "starter_code": "import numpy as np\n\n\ndef run(x):\n    raise NotImplementedError\n",
        "hints": [
            "First write the tiny example by hand (2-3 data points), then match your code to it.",
            "Do not import sklearn/torch - the point is the update rule.",
        ],
        "difficulty": max(2, min(5, 1 + level_idx)),
        "est_minutes": 30,
        "kind": "code" if level_idx >= 2 else "concept",
        "citation": anchor.citation,
    }
