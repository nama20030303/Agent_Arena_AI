import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api";
import NodeGraph from "../components/NodeGraph";
import { MapData } from "../types";

export default function MapScreen() {
  const [data, setData] = useState<MapData | null>(null);
  const [error, setError] = useState("");
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();

  const load = () => {
    api
      .get<MapData>("/game/map")
      .then(setData)
      .catch((e: Error) => setError(e.message));
  };
  useEffect(load, []);

  const worldCode = params.get("world") || data?.worlds[0]?.code || "python";
  const world = data?.worlds.find((w) => w.code === worldCode) || data?.worlds[0];

  const worldNodes = useMemo(() => (data ? data.nodes.filter((n) => n.world === world?.code) : []), [data, world]);
  const mastered = worldNodes.filter((n) => n.status === "mastered").length;

  if (error)
    return (
      <div className="screen">
        <div className="error-state">
          <p>{error}</p>
          <button className="btn btn-ghost" onClick={load}>Попробовать ещё раз</button>
        </div>
      </div>
    );
  if (!data || !world) return <div className="screen-loading">Карта знаний проявляется…</div>;

  return (
    <div className="screen map-screen" style={{ ["--accent" as string]: world.color }}>
      <div className="world-tabs">
        {data.worlds.map((w) => (
          <button
            key={w.code}
            className={`world-tab ${w.code === world.code ? "active" : ""}`}
            style={{ ["--accent" as string]: w.color }}
            onClick={() => setParams({ world: w.code })}
          >
            <span>{w.glyph}</span> {w.name}
          </button>
        ))}
      </div>
      <div className="map-head">
        <h2 className="map-title">{world.name}</h2>
        <p className="map-tagline">{world.tagline}</p>
        <div className="map-progress">
          Освоено {mastered} из {worldNodes.length}
          <div className="map-progress-bar">
            <div className="map-progress-fill" style={{ width: `${(mastered / Math.max(1, worldNodes.length)) * 100}%` }} />
          </div>
        </div>
      </div>
      <div className="map-canvas">
        <NodeGraph nodes={worldNodes} edges={data.edges} color={world.color} onSelect={(code) => navigate(`/node/${code}`)} />
      </div>
      <div className="map-legend">
        <span><i className="dot dot-available" /> доступен</span>
        <span><i className="dot dot-active" /> в процессе</span>
        <span><i className="dot dot-mastered" /> освоен</span>
        <span><i className="dot dot-locked" /> закрыт</span>
      </div>
    </div>
  );
}
