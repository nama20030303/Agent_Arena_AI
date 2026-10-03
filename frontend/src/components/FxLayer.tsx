import { useGame } from "../store";

/** Floating reward toasts + full-screen level-up moment. */
export default function FxLayer() {
  const { toasts, dismissToast, levelUp, clearLevelUp } = useGame();
  return (
    <>
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <button key={t.id} className={`toast toast-${t.kind}`} onClick={() => dismissToast(t.id)}>
            <span className="toast-glyph">{t.glyph}</span>
            <span>{t.text}</span>
          </button>
        ))}
      </div>
      {levelUp && (
        <div className="levelup-overlay" onClick={clearLevelUp} role="dialog" aria-label="Новый уровень">
          <div className="levelup-burst" />
          <div className="levelup-card">
            <div className="levelup-label">Новый уровень</div>
            <div className="levelup-number">{levelUp.to}</div>
            <div className="levelup-rank">{levelUp.rank}</div>
            <button className="btn btn-primary" onClick={clearLevelUp}>
              Продолжить путь
            </button>
          </div>
        </div>
      )}
    </>
  );
}
