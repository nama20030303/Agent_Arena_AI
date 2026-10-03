import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import Markdownish from "../components/Markdownish";
import ChestModal from "../components/ChestModal";
import { useGame } from "../store";
import { AnswerResult, NodeDetail, NodeQuestion } from "../types";

type Tab = "theory" | "practice";

export default function NodeScreen() {
  const { code } = useParams<{ code: string }>();
  const navigate = useNavigate();
  const { pushFx, refresh } = useGame();

  const [node, setNode] = useState<NodeDetail | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("theory");
  const [activeQ, setActiveQ] = useState<number | null>(null);
  const [option, setOption] = useState<number | null>(null);
  const [answer, setAnswer] = useState("");
  const [hints, setHints] = useState<string[]>([]);
  const [feedback, setFeedback] = useState<{ correct: boolean; explanation: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [masteredBanner, setMasteredBanner] = useState(false);
  const [chestToOpen, setChestToOpen] = useState<number | null>(null);
  const pendingChest = useRef<number | null>(null);

  const load = async () => {
    try {
      const d = await api.get<NodeDetail>(`/game/node/${code}`);
      setNode(d);
      if (d.status === "active" || d.progress > 0) setTab("practice");
      const firstUnsolved = d.questions.find((q) => !q.solved);
      setActiveQ(firstUnsolved ? firstUnsolved.id : null);
    } catch (e: unknown) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    setNode(null);
    setFeedback(null);
    setHints([]);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code]);

  if (error)
    return (
      <div className="screen">
        <div className="error-state">
          <p>{error}</p>
          <button className="btn btn-ghost" onClick={() => navigate(-1)}>Назад</button>
        </div>
      </div>
    );
  if (!node) return <div className="screen-loading">Узел знаний активируется…</div>;

  const q = node.questions.find((x) => x.id === activeQ) || null;
  const solvedCount = node.questions.filter((x) => x.solved).length;

  const submit = async () => {
    if (!q || busy) return;
    setBusy(true);
    setFeedback(null);
    try {
      const res = await api.post<AnswerResult>("/game/answer", {
        question_id: q.id,
        answer,
        option,
      });
      setFeedback({ correct: res.correct, explanation: res.explanation });
      pushFx(res.fx, { unlocked: res.node.unlocked });
      if (navigator.vibrate && res.correct) navigator.vibrate(res.node.mastered ? [50, 40, 90] : 25);
      if (res.node.mastered) {
        setMasteredBanner(true);
        const granted = res.fx.chests_granted || [];
        if (granted.length > 0) pendingChest.current = granted[granted.length - 1].id;
      }
      const d = await api.get<NodeDetail>(`/game/node/${code}`);
      setNode(d);
      refresh();
      if (res.correct) {
        setTimeout(() => {
          setFeedback(null);
          setOption(null);
          setAnswer("");
          setHints([]);
          const next = d.questions.find((x) => !x.solved);
          setActiveQ(next ? next.id : null);
        }, res.node.mastered ? 600 : 1600);
      }
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const takeHint = async () => {
    if (!q) return;
    try {
      const res = await api.post<{ hint: string }>("/game/hint", { question_id: q.id });
      setHints((prev) => (prev.includes(res.hint) ? prev : [...prev, res.hint]));
    } catch {
      /* no more hints */
    }
  };

  const accent = node.world.color;

  return (
    <div className="screen node-screen" style={{ ["--accent" as string]: accent }}>
      <header className="node-head">
        <button className="back-btn" onClick={() => navigate(`/map?world=${node.world.code}`)} aria-label="Назад к карте">
          ←
        </button>
        <div className="node-head-info">
          <div className="node-world">
            {node.world.glyph} {node.world.name}
            {node.is_boss && <span className="boss-chip">БОСС</span>}
          </div>
          <h2 className="node-name">{node.name}</h2>
        </div>
        <div className="node-progress-ring">
          <span>{Math.round(node.progress)}%</span>
        </div>
      </header>

      {node.status === "mastered" && (
        <div className="mastered-banner">
          <span className="mastered-glyph">✓</span> Концепт освоен · +{node.xp_reward} XP получено
        </div>
      )}

      <div className="node-tabs">
        <button className={`node-tab ${tab === "theory" ? "active" : ""}`} onClick={() => setTab("theory")}>
          Теория
        </button>
        <button className={`node-tab ${tab === "practice" ? "active" : ""}`} onClick={() => setTab("practice")}>
          Практика · {solvedCount}/{node.questions.length}
        </button>
      </div>

      {tab === "theory" && (
        <div className="theory-pane">
          <Markdownish text={node.theory} />
          <div className="theory-actions">
            <button className="btn btn-primary btn-block" onClick={() => setTab("practice")}>
              {node.is_boss ? "Начать испытание" : "Перейти к практике"}
            </button>
            <button
              className="btn btn-ghost btn-block"
              onClick={() => navigate(`/teacher?node=${node.code}`)}
            >
              Спросить Учителя
            </button>
          </div>
        </div>
      )}

      {tab === "practice" && (
        <div className="practice-pane">
          <div className="q-dots">
            {node.questions.map((x) => (
              <button
                key={x.id}
                className={`q-dot ${x.solved ? "solved" : ""} ${x.id === activeQ ? "current" : ""}`}
                onClick={() => {
                  setActiveQ(x.id);
                  setFeedback(null);
                  setOption(null);
                  setAnswer("");
                  setHints([]);
                }}
                aria-label={x.solved ? "Решено" : "Задание"}
              >
                {x.solved ? "✓" : "·"}
              </button>
            ))}
          </div>

          {q === null && (
            <div className="practice-done">
              <div className="practice-done-glyph">✦</div>
              <h3>Все задания решены</h3>
              <p>Этот узел знаний полностью освоен. Карта ждёт твоего следующего шага.</p>
              <button className="btn btn-primary" onClick={() => navigate(`/map?world=${node.world.code}`)}>
                Вернуться к карте
              </button>
            </div>
          )}

          {q !== null && (
            <QuestionCard
              key={q.id}
              q={q}
              option={option}
              setOption={setOption}
              answer={answer}
              setAnswer={setAnswer}
              hints={hints}
              takeHint={takeHint}
              feedback={feedback}
              busy={busy}
              submit={submit}
              retry={() => setFeedback(null)}
              askTeacher={() => navigate(`/teacher?node=${node.code}`)}
            />
          )}
        </div>
      )}

      {masteredBanner && (
        <div className="modal-backdrop" onClick={() => setMasteredBanner(false)}>
          <div className="mastered-modal" onClick={(e) => e.stopPropagation()}>
            <div className="mastered-rays" />
            <div className="mastered-title">КОНЦЕПТ ОСВОЕН</div>
            <div className="mastered-node">{node.name}</div>
            <div className="mastered-sub">Узел знаний активирован</div>
            <button
              className="btn btn-primary"
              onClick={() => {
                setMasteredBanner(false);
                if (pendingChest.current !== null) {
                  setChestToOpen(pendingChest.current);
                  pendingChest.current = null;
                }
              }}
            >
              {pendingChest.current !== null ? "Забрать награду" : "Продолжить"}
            </button>
          </div>
        </div>
      )}

      {chestToOpen !== null && (
        <ChestModal
          chestId={chestToOpen}
          onClose={() => {
            setChestToOpen(null);
            refresh();
          }}
        />
      )}
    </div>
  );
}

function QuestionCard(props: {
  q: NodeQuestion;
  option: number | null;
  setOption: (v: number | null) => void;
  answer: string;
  setAnswer: (v: string) => void;
  hints: string[];
  takeHint: () => void;
  feedback: { correct: boolean; explanation: string } | null;
  busy: boolean;
  submit: () => void;
  retry: () => void;
  askTeacher: () => void;
}) {
  const { q, option, setOption, answer, setAnswer, hints, takeHint, feedback, busy, submit, retry, askTeacher } = props;
  const canSubmit = q.type === "mcq" ? option !== null : answer.trim().length > 0;

  return (
    <div className={`question-card ${feedback ? (feedback.correct ? "ok" : "fail") : ""}`}>
      <div className="q-meta">
        <span className="q-difficulty" aria-label={`Сложность ${q.difficulty} из 5`}>
          {"◆".repeat(q.difficulty)}
          {"◇".repeat(Math.max(0, 5 - q.difficulty))}
        </span>
        {q.solved && <span className="q-solved-chip">решено ранее</span>}
      </div>
      <pre className="q-prompt">{q.prompt}</pre>

      {q.type === "mcq" ? (
        <div className="q-options">
          {q.options.map((opt, i) => (
            <button
              key={i}
              className={`q-option ${option === i ? "picked" : ""}`}
              onClick={() => !feedback && setOption(i)}
              disabled={!!feedback && feedback.correct}
            >
              <span className="q-option-letter">{"АБВГД"[i]}</span>
              {opt}
            </button>
          ))}
        </div>
      ) : (
        <input
          className="q-input"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="Твой ответ…"
          disabled={!!feedback && feedback.correct}
          onKeyDown={(e) => e.key === "Enter" && canSubmit && !feedback && submit()}
        />
      )}

      {hints.map((h, i) => (
        <div key={i} className="hint-bubble">
          <span className="hint-glyph">💡</span> {h}
        </div>
      ))}

      {feedback && (
        <div className={`q-feedback ${feedback.correct ? "ok" : "fail"}`}>
          <div className="q-feedback-head">{feedback.correct ? "Верно!" : "Пока не так"}</div>
          {feedback.explanation && <div className="q-feedback-body">{feedback.explanation}</div>}
        </div>
      )}

      <div className="q-actions">
        {!feedback && (
          <>
            {q.hints_total > 0 && hints.length < q.hints_total && (
              <button className="btn btn-ghost" onClick={takeHint}>
                Подсказка ({hints.length}/{q.hints_total})
              </button>
            )}
            <button className="btn btn-primary" disabled={!canSubmit || busy} onClick={submit}>
              {busy ? "Проверяем…" : "Ответить"}
            </button>
          </>
        )}
        {feedback && !feedback.correct && (
          <>
            <button className="btn btn-ghost" onClick={askTeacher}>
              Спросить Учителя
            </button>
            <button className="btn btn-primary" onClick={retry}>
              Попробовать ещё раз
            </button>
          </>
        )}
      </div>
    </div>
  );
}
