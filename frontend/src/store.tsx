import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api, getToken } from "./api";
import { Fx, GameState } from "./types";

export interface FxToast {
  id: number;
  kind: "xp" | "achievement" | "quest" | "chest" | "streak" | "info" | "unlock";
  text: string;
  glyph?: string;
}

interface GameCtx {
  ready: boolean;
  authed: boolean;
  game: GameState | null;
  refresh: () => Promise<void>;
  setAuthed: (v: boolean) => void;
  pushFx: (fx: Fx | undefined, extra?: { unlocked?: string[] }) => void;
  toasts: FxToast[];
  dismissToast: (id: number) => void;
  levelUp: { to: number; rank: string } | null;
  clearLevelUp: () => void;
}

const Ctx = createContext<GameCtx>(null as unknown as GameCtx);

let toastSeq = 1;

export function GameProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(!!getToken());
  const [game, setGame] = useState<GameState | null>(null);
  const [toasts, setToasts] = useState<FxToast[]>([]);
  const [levelUp, setLevelUp] = useState<{ to: number; rank: string } | null>(null);
  const timers = useRef<number[]>([]);

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setReady(true);
      return;
    }
    try {
      const data = await api.get<GameState>("/game/state");
      setGame(data);
      setAuthed(true);
    } catch (e: unknown) {
      const status = (e as { status?: number }).status;
      if (status === 401) setAuthed(false);
    } finally {
      setReady(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    return () => timers.current.forEach((t) => window.clearTimeout(t));
  }, [refresh]);

  const addToast = useCallback((toast: Omit<FxToast, "id">, delay: number) => {
    const id = toastSeq++;
    const t = window.setTimeout(() => {
      setToasts((prev) => [...prev, { ...toast, id }]);
      const t2 = window.setTimeout(() => {
        setToasts((prev) => prev.filter((x) => x.id !== id));
      }, 3400);
      timers.current.push(t2);
    }, delay);
    timers.current.push(t);
  }, []);

  const pushFx = useCallback(
    (fx: Fx | undefined, extra?: { unlocked?: string[] }) => {
      if (!fx) return;
      let delay = 120;
      const step = 650;
      if (fx.xp_gained) {
        addToast({ kind: "xp", text: `+${fx.xp_gained} XP`, glyph: "✦" }, delay);
        delay += step;
      }
      if (extra?.unlocked?.length) {
        addToast({ kind: "unlock", text: "Открыт новый узел знаний", glyph: "◈" }, delay);
        delay += step;
      }
      (fx.quests_completed || []).forEach((q) => {
        addToast({ kind: "quest", text: `Задание выполнено: ${q.title}`, glyph: "✔" }, delay);
        delay += step;
      });
      (fx.achievements || []).forEach((a) => {
        addToast({ kind: "achievement", text: `Достижение: ${a.name}`, glyph: a.glyph }, delay);
        delay += step;
      });
      (fx.chests_granted || []).forEach(() => {
        addToast({ kind: "chest", text: "Получен сундук!", glyph: "⬢" }, delay);
        delay += step;
      });
      if (fx.streak && (fx.streak.kind === "extended" || fx.streak.kind === "shield_used")) {
        addToast({ kind: "streak", text: `Серия: ${fx.streak.streak} дн.`, glyph: "⚡" }, delay);
        delay += step;
      }
      if (fx.level_up) {
        const lu = fx.level_up;
        const t = window.setTimeout(() => setLevelUp({ to: lu.to, rank: lu.rank }), delay + 200);
        timers.current.push(t);
      }
    },
    [addToast]
  );

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((x) => x.id !== id));
  }, []);

  return (
    <Ctx.Provider
      value={{
        ready,
        authed,
        game,
        refresh,
        setAuthed,
        pushFx,
        toasts,
        dismissToast,
        levelUp,
        clearLevelUp: () => setLevelUp(null),
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useGame() {
  return useContext(Ctx);
}
