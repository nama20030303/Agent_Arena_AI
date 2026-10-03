import { useState } from "react";
import { api } from "../api";
import { useGame } from "../store";
import { ChestOpenResult, RARITY_RU, KIND_RU } from "../types";

interface Props {
  chestId: number;
  onClose: (opened: boolean) => void;
}

type Phase = "closed" | "charging" | "reveal";

/** Chest-opening ceremony: anticipation → activation → rarity reveal → reward. */
export default function ChestModal({ chestId, onClose }: Props) {
  const { pushFx, refresh } = useGame();
  const [phase, setPhase] = useState<Phase>("closed");
  const [result, setResult] = useState<ChestOpenResult | null>(null);
  const [error, setError] = useState("");

  const open = async () => {
    if (phase !== "closed") return;
    setPhase("charging");
    try {
      const [res] = await Promise.all([
        api.post<ChestOpenResult>(`/game/chests/${chestId}/open`),
        new Promise((r) => setTimeout(r, 1500)),
      ]);
      setResult(res);
      setPhase("reveal");
      if (navigator.vibrate) navigator.vibrate(res.rarity === "legendary" || res.rarity === "mythic" ? [60, 40, 120] : 35);
      pushFx(res.fx);
      refresh();
    } catch (e: unknown) {
      setError((e as Error).message || "Не удалось открыть сундук.");
      setPhase("closed");
    }
  };

  return (
    <div className="modal-backdrop" onClick={() => phase !== "charging" && onClose(phase === "reveal")}>
      <div className="chest-stage" onClick={(e) => e.stopPropagation()}>
        {phase !== "reveal" && (
          <>
            <div className={`chest ${phase}`} onClick={open} role="button" aria-label="Открыть сундук">
              <div className="chest-lid" />
              <div className="chest-base" />
              <div className="chest-glow" />
            </div>
            <div className="chest-caption">
              {phase === "closed" ? "Коснись сундука, чтобы открыть" : "Сундук активируется…"}
            </div>
            {error && <div className="error-text">{error}</div>}
          </>
        )}
        {phase === "reveal" && result && (
          <div className={`reward rarity-${result.rarity}`}>
            <div className="reward-halo" />
            <div className="reward-glyph">{result.item.glyph}</div>
            <div className="reward-rarity">{RARITY_RU[result.rarity] || result.rarity}</div>
            <div className="reward-name">{result.item.name}</div>
            <div className="reward-kind">{KIND_RU[result.item.kind] || result.item.kind}</div>
            <div className="reward-desc">{result.item.description}</div>
            {result.duplicate && <div className="reward-dup">Повтор — предмет уже в коллекции</div>}
            <div className="reward-xp">+{result.xp_bonus} XP</div>
            <button className="btn btn-primary" onClick={() => onClose(true)}>
              В коллекцию
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
