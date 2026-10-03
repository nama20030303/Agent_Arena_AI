import { useState } from "react";
import { api, setToken } from "../api";
import { useGame } from "../store";
import { Fx, PlayerPublicState } from "../types";

const WORLD_CHOICES = [
  { code: "python", name: "Программирование", glyph: "⌘", color: "#22d3ee", desc: "Python: от первой строки до своих программ" },
  { code: "physics", name: "Физика", glyph: "⚛", color: "#a78bfa", desc: "Механика: силы, энергия, движение" },
  { code: "math", name: "Математика", glyph: "∑", color: "#f59e0b", desc: "Основы: от чисел до вероятности" },
];

const LEVELS = [
  { code: "beginner", name: "Начинаю с нуля", desc: "Почти ничего не знаю — и это отличная стартовая точка" },
  { code: "middle", name: "Кое-что знаю", desc: "Основы знакомы, хочу систематизировать и углубить" },
  { code: "advanced", name: "Уверенный уровень", desc: "Хочу сложные задачи и испытания" },
];

const INTENSITIES = [
  { code: "light", name: "Лёгкий темп", desc: "5–10 минут в день" },
  { code: "normal", name: "Обычный темп", desc: "15–25 минут в день" },
  { code: "deep", name: "Глубокое погружение", desc: "30+ минут в день" },
];

type Step = "login" | "interests" | "level" | "goal" | "intensity";

export default function Auth({ needsOnboarding }: { needsOnboarding: boolean }) {
  const { refresh, pushFx } = useGame();
  const [step, setStep] = useState<Step>(needsOnboarding ? "interests" : "login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [mode, setMode] = useState<"register" | "login">("register");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [interests, setInterests] = useState<string[]>([]);
  const [level, setLevel] = useState("beginner");
  const [goal, setGoal] = useState("");
  const [intensity, setIntensity] = useState("normal");

  const submitAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim()) return;
    setBusy(true);
    setError("");
    try {
      const path = mode === "register" ? "/auth/register" : "/auth/login";
      const body =
        mode === "register"
          ? { username: username.trim(), display_name: displayName.trim() || username.trim() }
          : { username: username.trim() };
      const res = await api.post<{ token: string }>(path, body);
      setToken(res.token);
      const state = await api.get<{ state: PlayerPublicState }>("/game/state");
      if (state.state.onboarded) {
        await refresh();
      } else {
        setStep("interests");
      }
    } catch (err: unknown) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const finish = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await api.post<{ fx: Fx }>("/game/onboard", {
        interests,
        start_level: level,
        goal,
        intensity,
      });
      await refresh();
      pushFx(res.fx);
    } catch (err: unknown) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  if (step === "login") {
    return (
      <div className="auth-screen">
        <div className="auth-hero">
          <div className="auth-emblem">✦</div>
          <h1 className="auth-title">Вселенная Знаний</h1>
          <p className="auth-sub">Учись как в игре. Становись сильнее по-настоящему.</p>
        </div>
        <form className="auth-form" onSubmit={submitAuth}>
          <label className="field">
            <span>Имя профиля</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="например, alisa"
              autoFocus
              maxLength={40}
            />
          </label>
          {mode === "register" && (
            <label className="field">
              <span>Как к тебе обращаться?</span>
              <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="Алиса" maxLength={60} />
            </label>
          )}
          {error && <div className="error-text">{error}</div>}
          <button className="btn btn-primary btn-block" disabled={busy || !username.trim()}>
            {busy ? "Входим…" : mode === "register" ? "Начать путь" : "Продолжить путь"}
          </button>
          <button
            type="button"
            className="btn-link"
            onClick={() => {
              setMode(mode === "register" ? "login" : "register");
              setError("");
            }}
          >
            {mode === "register" ? "У меня уже есть профиль" : "Создать новый профиль"}
          </button>
        </form>
      </div>
    );
  }

  const stepIndex = { interests: 1, level: 2, goal: 3, intensity: 4 }[step];

  return (
    <div className="auth-screen onboarding">
      <div className="ob-progress">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className={`ob-dot ${i <= stepIndex ? "on" : ""}`} />
        ))}
      </div>

      {step === "interests" && (
        <div className="ob-step">
          <h2>Что тебя зовёт?</h2>
          <p className="ob-sub">Выбери миры, которые хочешь исследовать. Можно несколько.</p>
          <div className="ob-cards">
            {WORLD_CHOICES.map((w) => (
              <button
                key={w.code}
                className={`ob-card ${interests.includes(w.code) ? "selected" : ""}`}
                style={{ ["--accent" as string]: w.color }}
                onClick={() =>
                  setInterests((prev) => (prev.includes(w.code) ? prev.filter((x) => x !== w.code) : [...prev, w.code]))
                }
              >
                <span className="ob-card-glyph">{w.glyph}</span>
                <span className="ob-card-name">{w.name}</span>
                <span className="ob-card-desc">{w.desc}</span>
              </button>
            ))}
          </div>
          <button className="btn btn-primary btn-block" disabled={interests.length === 0} onClick={() => setStep("level")}>
            Дальше
          </button>
        </div>
      )}

      {step === "level" && (
        <div className="ob-step">
          <h2>Твой уровень сейчас</h2>
          <p className="ob-sub">Честный ответ поможет подобрать правильную сложность.</p>
          <div className="ob-cards">
            {LEVELS.map((l) => (
              <button key={l.code} className={`ob-card ${level === l.code ? "selected" : ""}`} onClick={() => setLevel(l.code)}>
                <span className="ob-card-name">{l.name}</span>
                <span className="ob-card-desc">{l.desc}</span>
              </button>
            ))}
          </div>
          <button className="btn btn-primary btn-block" onClick={() => setStep("goal")}>
            Дальше
          </button>
        </div>
      )}

      {step === "goal" && (
        <div className="ob-step">
          <h2>Зачем ты здесь?</h2>
          <p className="ob-sub">Цель можно менять в любой момент.</p>
          <textarea
            className="ob-goal"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Например: научиться программировать и написать свою первую игру"
            rows={3}
            maxLength={300}
          />
          <button className="btn btn-primary btn-block" onClick={() => setStep("intensity")}>
            Дальше
          </button>
        </div>
      )}

      {step === "intensity" && (
        <div className="ob-step">
          <h2>Выбери темп</h2>
          <p className="ob-sub">Пропущенный день не разрушит прогресс — у тебя есть защита серии.</p>
          <div className="ob-cards">
            {INTENSITIES.map((i) => (
              <button key={i.code} className={`ob-card ${intensity === i.code ? "selected" : ""}`} onClick={() => setIntensity(i.code)}>
                <span className="ob-card-name">{i.name}</span>
                <span className="ob-card-desc">{i.desc}</span>
              </button>
            ))}
          </div>
          {error && <div className="error-text">{error}</div>}
          <button className="btn btn-primary btn-block" disabled={busy} onClick={finish}>
            {busy ? "Создаём вселенную…" : "Войти во вселенную"}
          </button>
        </div>
      )}
    </div>
  );
}
