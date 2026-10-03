import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGame } from "../store";
import LevelRing from "../components/LevelRing";
import ChestModal from "../components/ChestModal";

export default function Home() {
  const { game, refresh } = useGame();
  const navigate = useNavigate();
  const [chestId, setChestId] = useState<number | null>(null);

  if (!game) return <div className="screen-loading">Вселенная загружается…</div>;

  const { state, objective, quests, chests, worlds, display_name } = game;
  const hour = new Date().getHours();
  const greeting = hour < 5 ? "Доброй ночи" : hour < 12 ? "Доброе утро" : hour < 18 ? "Добрый день" : "Добрый вечер";

  return (
    <div className="screen home">
      <header className="home-header">
        <div className="home-id">
          <LevelRing level={state.level} progress={state.xp_for_next ? state.xp_into_level / state.xp_for_next : 0} />
          <div>
            <div className="home-greeting">{greeting},</div>
            <div className="home-name">{display_name}</div>
            <div className="home-rank">
              {state.rank}
              {state.selected_title && <span className="home-title-chip">{titleName(state.selected_title)}</span>}
            </div>
          </div>
        </div>
        <div className="home-meta">
          <div className="meta-pill" title="Серия дней">
            <span className="meta-glyph streak">⚡</span> {state.streak}
          </div>
          <div className="meta-pill" title="Сила знаний">
            <span className="meta-glyph power">✦</span> {state.knowledge_power}
          </div>
        </div>
      </header>

      <div className="xp-line">
        <div className="xp-bar">
          <div
            className="xp-fill"
            style={{ width: `${Math.min(100, (state.xp_into_level / Math.max(1, state.xp_for_next)) * 100)}%` }}
          />
        </div>
        <div className="xp-caption">
          {state.xp_into_level} / {state.xp_for_next} XP до {state.level + 1} уровня
        </div>
      </div>

      {objective && (
        <button
          className={`objective-card ${objective.is_boss ? "boss" : ""}`}
          style={{ ["--accent" as string]: objective.color }}
          onClick={() => navigate(`/node/${objective.node}`)}
        >
          <div className="objective-tag">{objective.is_boss ? "Босс-испытание" : "Текущая цель"}</div>
          <div className="objective-name">{objective.name}</div>
          <div className="objective-world">
            {objective.world_name} · {objective.short}
          </div>
          {objective.progress > 0 && (
            <div className="objective-progress">
              <div className="objective-progress-fill" style={{ width: `${objective.progress}%` }} />
            </div>
          )}
          <div className="objective-cta">{objective.progress > 0 ? "Продолжить обучение →" : "Начать →"}</div>
        </button>
      )}

      {chests.length > 0 && (
        <section className="section">
          <h3 className="section-title">Награды ждут</h3>
          <div className="chest-shelf">
            {chests.map((c) => (
              <button key={c.id} className="chest-slot" onClick={() => setChestId(c.id)} aria-label="Открыть сундук">
                <span className="chest-icon">⬢</span>
                <span className="chest-source">{chestSource(c.source)}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="section">
        <h3 className="section-title">Задания дня</h3>
        <div className="quest-list">
          {quests.map((q) => (
            <div key={q.code} className={`quest-card ${q.completed ? "done" : ""}`}>
              <div className="quest-check">{q.completed ? "✓" : ""}</div>
              <div className="quest-body">
                <div className="quest-title">{q.title}</div>
                <div className="quest-progress">
                  <div className="quest-progress-fill" style={{ width: `${(q.progress / q.target) * 100}%` }} />
                </div>
              </div>
              <div className="quest-reward">
                +{q.xp} XP{q.chest && <span className="quest-chest" title="Сундук за выполнение">⬢</span>}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h3 className="section-title">Твои миры</h3>
        <div className="world-list">
          {worlds.map((w) => (
            <button
              key={w.code}
              className="world-card"
              style={{ ["--accent" as string]: w.color }}
              onClick={() => navigate(`/map?world=${w.code}`)}
            >
              <span className="world-glyph">{w.glyph}</span>
              <span className="world-info">
                <span className="world-name">{w.name}</span>
                <span className="world-stat">
                  {w.mastered} / {w.total} концептов
                </span>
              </span>
              <span className="world-power">{w.power}</span>
            </button>
          ))}
        </div>
      </section>

      {chestId !== null && (
        <ChestModal
          chestId={chestId}
          onClose={() => {
            setChestId(null);
            refresh();
          }}
        />
      )}
    </div>
  );
}

function chestSource(source: string): string {
  const map: Record<string, string> = {
    node: "За концепт",
    boss: "За босса",
    quest: "За задание",
    level: "За уровень",
    onboarding: "Первый шаг",
  };
  return map[source] || "Награда";
}

function titleName(code: string): string {
  const map: Record<string, string> = {
    "t-seeker": "Искатель истины",
    "t-nightowl": "Ночной мыслитель",
    "t-architect": "Архитектор знаний",
    "t-polymath": "Полимат",
  };
  return map[code] || code;
}
