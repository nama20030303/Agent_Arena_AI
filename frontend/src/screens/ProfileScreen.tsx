import { useEffect, useState } from "react";
import { api, clearToken } from "../api";
import { useGame } from "../store";
import LevelRing from "../components/LevelRing";
import { Achievement, ItemInfo, LeaderboardEntry, ProfileData, RARITY_RU, KIND_RU } from "../types";

type Tab = "achievements" | "inventory" | "history" | "rating";

export default function ProfileScreen() {
  const { refresh, setAuthed } = useGame();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [items, setItems] = useState<ItemInfo[]>([]);
  const [leaders, setLeaders] = useState<LeaderboardEntry[]>([]);
  const [tab, setTab] = useState<Tab>("achievements");
  const [error, setError] = useState("");

  const load = () => {
    Promise.all([
      api.get<ProfileData>("/game/profile"),
      api.get<{ items: ItemInfo[] }>("/game/inventory"),
      api.get<{ entries: LeaderboardEntry[] }>("/game/leaderboard"),
    ])
      .then(([p, inv, lb]) => {
        setProfile(p);
        setItems(inv.items);
        setLeaders(lb.entries);
      })
      .catch((e: Error) => setError(e.message));
  };
  useEffect(load, []);

  if (error)
    return (
      <div className="screen">
        <div className="error-state">
          <p>{error}</p>
          <button className="btn btn-ghost" onClick={load}>Попробовать ещё раз</button>
        </div>
      </div>
    );
  if (!profile) return <div className="screen-loading">Профиль собирается…</div>;

  const s = profile.state;
  const accuracy = s.answers_total > 0 ? Math.round((100 * s.answers_correct) / s.answers_total) : 0;

  const equipTitle = async (code: string) => {
    await api.post("/game/equip", { title: s.selected_title === code ? "" : code });
    load();
    refresh();
  };

  return (
    <div className="screen profile-screen">
      <header className="profile-head">
        <div className={`profile-avatar ${s.selected_frame ? "framed" : ""}`}>
          <span>{profile.display_name.slice(0, 1).toUpperCase()}</span>
        </div>
        <div className="profile-id">
          <div className="profile-name">{profile.display_name}</div>
          <div className="profile-rank">
            {s.rank} · {s.level} уровень
          </div>
          {s.selected_title && <div className="profile-title-chip">{titleOf(items, s.selected_title)}</div>}
        </div>
        <LevelRing level={s.level} progress={s.xp_for_next ? s.xp_into_level / s.xp_for_next : 0} size={56} />
      </header>

      <div className="stat-grid">
        <div className="stat-card"><div className="stat-value power">✦ {s.knowledge_power}</div><div className="stat-label">Сила знаний</div></div>
        <div className="stat-card"><div className="stat-value streak">⚡ {s.streak}</div><div className="stat-label">Серия (рекорд {s.best_streak})</div></div>
        <div className="stat-card"><div className="stat-value">{accuracy}%</div><div className="stat-label">Точность</div></div>
        <div className="stat-card"><div className="stat-value">♛ {s.bosses_defeated}</div><div className="stat-label">Боссы</div></div>
      </div>

      <section className="section">
        <h3 className="section-title">Сила по мирам</h3>
        {profile.worlds.map((w) => (
          <div key={w.code} className="skill-row" style={{ ["--accent" as string]: w.color }}>
            <span className="skill-glyph">{w.glyph}</span>
            <span className="skill-name">{w.name}</span>
            <div className="skill-bar"><div className="skill-fill" style={{ width: `${w.power}%` }} /></div>
            <span className="skill-value">{w.power}</span>
          </div>
        ))}
      </section>

      <div className="node-tabs profile-tabs">
        {([
          ["achievements", "Достижения"],
          ["inventory", "Коллекция"],
          ["history", "Путь"],
          ["rating", "Рейтинг"],
        ] as [Tab, string][]).map(([t, label]) => (
          <button key={t} className={`node-tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {label}
          </button>
        ))}
      </div>

      {tab === "achievements" && <Achievements list={profile.achievements} />}

      {tab === "inventory" && (
        <div className="inventory-grid">
          {items.length === 0 && (
            <div className="empty-state">Коллекция пока пуста — сундуки ждут своих открывателей.</div>
          )}
          {items.map((it) => (
            <button
              key={it.code}
              className={`inv-item rarity-${it.rarity} ${it.kind === "title" ? "equipable" : ""} ${s.selected_title === it.code ? "equipped" : ""}`}
              onClick={() => it.kind === "title" && equipTitle(it.code)}
              title={it.description}
            >
              <span className="inv-glyph">{it.glyph}</span>
              <span className="inv-name">{it.name}</span>
              <span className="inv-rarity">{RARITY_RU[it.rarity]} · {KIND_RU[it.kind]}</span>
              {it.count && it.count > 1 && <span className="inv-count">×{it.count}</span>}
              {it.kind === "title" && <span className="inv-equip">{s.selected_title === it.code ? "надет" : "надеть"}</span>}
            </button>
          ))}
        </div>
      )}

      {tab === "history" && (
        <div className="timeline">
          {profile.timeline.length === 0 && <div className="empty-state">История пока не началась. Первый шаг — за тобой.</div>}
          {profile.timeline.map((t, i) => (
            <div key={i} className="timeline-row">
              <div className={`timeline-dot tl-${t.type.replace(".", "-")}`} />
              <div className="timeline-body">
                <span className="timeline-label">{t.label}</span>
                {t.detail && <span className="timeline-detail">{t.detail}</span>}
              </div>
              <div className="timeline-date">{formatDate(t.at)}</div>
            </div>
          ))}
        </div>
      )}

      {tab === "rating" && (
        <div className="leaderboard">
          {leaders.map((l, i) => (
            <div key={l.username} className={`lb-row ${l.me ? "me" : ""}`}>
              <span className="lb-place">{i + 1}</span>
              <span className="lb-name">{l.display_name}</span>
              <span className="lb-rank">{l.rank}</span>
              <span className="lb-xp">{l.xp} XP</span>
            </div>
          ))}
        </div>
      )}

      <button
        className="btn-link logout"
        onClick={() => {
          clearToken();
          setAuthed(false);
          window.location.href = "/";
        }}
      >
        Выйти из профиля
      </button>
    </div>
  );
}

function Achievements({ list }: { list: Achievement[] }) {
  const unlocked = list.filter((a) => a.unlocked);
  const locked = list.filter((a) => !a.unlocked);
  return (
    <div className="ach-grid">
      {[...unlocked, ...locked].map((a) => (
        <div key={a.code} className={`ach-card ${a.unlocked ? "unlocked" : "locked"}`} title={a.description}>
          <span className="ach-glyph">{a.glyph}</span>
          <span className="ach-name">{a.name}</span>
          <span className="ach-desc">{a.description}</span>
        </div>
      ))}
    </div>
  );
}

function titleOf(items: ItemInfo[], code: string): string {
  const item = items.find((i) => i.code === code);
  return item ? item.name.replace("Титул: ", "") : code;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}
