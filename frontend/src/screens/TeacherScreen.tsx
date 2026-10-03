import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { useGame } from "../store";
import { ChatMessage, Fx } from "../types";
import Markdownish from "../components/Markdownish";

const MODES = [
  { code: "explain", name: "Объяснение" },
  { code: "socratic", name: "Сократ" },
  { code: "practice", name: "Практика" },
  { code: "challenge", name: "Вызов" },
  { code: "review", name: "Повторение" },
];

const SUGGESTIONS = [
  "Объясни, что такое переменные",
  "Почему ракета летит в космосе?",
  "Как решать квадратные уравнения?",
  "Что такое вероятность?",
];

export default function TeacherScreen() {
  const [params] = useSearchParams();
  const nodeCode = params.get("node") || "";
  const { pushFx, refresh } = useGame();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("explain");
  const [busy, setBusy] = useState(false);
  const [typing, setTyping] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const typeTimer = useRef<number | null>(null);

  useEffect(() => {
    api.get<{ messages: ChatMessage[] }>("/game/teacher/history").then((d) => setMessages(d.messages));
    return () => {
      if (typeTimer.current) window.clearInterval(typeTimer.current);
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, typing, busy]);

  const send = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || busy) return;
    setInput("");
    setBusy(true);
    setMessages((prev) => [...prev, { role: "user", text: msg, mode, node: nodeCode, via_ai: false }]);
    try {
      const res = await api.post<{ text: string; via_ai: boolean; mode: string; fx: Fx }>("/game/teacher", {
        message: msg,
        node_code: nodeCode,
        mode,
      });
      pushFx(res.fx);
      refresh();
      // progressive reveal — the teacher "thinks out" the answer
      const full = res.text;
      let i = 0;
      setTyping("");
      typeTimer.current = window.setInterval(() => {
        i += Math.max(2, Math.round(full.length / 90));
        if (i >= full.length) {
          if (typeTimer.current) window.clearInterval(typeTimer.current);
          setTyping("");
          setMessages((prev) => [...prev, { role: "teacher", text: full, mode: res.mode, node: nodeCode, via_ai: res.via_ai }]);
          setBusy(false);
        } else {
          setTyping(full.slice(0, i));
        }
      }, 28);
    } catch (e: unknown) {
      setMessages((prev) => [
        ...prev,
        { role: "teacher", text: (e as Error).message || "Не получилось ответить. Попробуем ещё раз.", mode, node: nodeCode, via_ai: false },
      ]);
      setBusy(false);
    }
  };

  return (
    <div className="screen teacher-screen">
      <header className="teacher-head">
        <div className="teacher-avatar">
          <div className="teacher-core" />
        </div>
        <div>
          <div className="teacher-name">Учитель</div>
          <div className="teacher-status">{busy ? "размышляет…" : "готов к диалогу"}</div>
        </div>
        {nodeCode && <div className="teacher-context">контекст: {nodeCode}</div>}
      </header>

      <div className="mode-chips">
        {MODES.map((m) => (
          <button key={m.code} className={`mode-chip ${mode === m.code ? "active" : ""}`} onClick={() => setMode(m.code)}>
            {m.name}
          </button>
        ))}
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 && !busy && (
          <div className="chat-empty">
            <p>Эта часть вселенной ещё не исследована. Задай Учителю первый вопрос — любой.</p>
            <div className="chat-suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="suggestion" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            {m.role === "teacher" ? <Markdownish text={m.text} /> : <p>{m.text}</p>}
            {m.role === "teacher" && !m.via_ai && <div className="offline-badge">автономный режим</div>}
          </div>
        ))}
        {busy && (
          <div className="chat-msg teacher">
            {typing ? <Markdownish text={typing} /> : <div className="typing-dots"><span /><span /><span /></div>}
          </div>
        )}
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Спроси о чём угодно…"
          rows={1}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button className="chat-send" onClick={() => send()} disabled={busy || !input.trim()} aria-label="Отправить">
          ↑
        </button>
      </div>
    </div>
  );
}
