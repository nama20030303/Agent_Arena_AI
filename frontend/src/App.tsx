import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { GameProvider, useGame } from "./store";
import Starfield from "./components/Starfield";
import FxLayer from "./components/FxLayer";
import Auth from "./screens/Auth";
import Home from "./screens/Home";
import MapScreen from "./screens/MapScreen";
import NodeScreen from "./screens/NodeScreen";
import TeacherScreen from "./screens/TeacherScreen";
import ProfileScreen from "./screens/ProfileScreen";

function Shell() {
  const { ready, authed, game } = useGame();
  const location = useLocation();

  if (!ready) {
    return (
      <div className="boot">
        <div className="boot-core" />
        <div className="boot-text">Вселенная Знаний</div>
      </div>
    );
  }

  if (!authed || !game) {
    return (
      <div className="app-frame">
        <Auth needsOnboarding={false} />
        <FxLayer />
      </div>
    );
  }

  if (!game.state.onboarded) {
    return (
      <div className="app-frame">
        <Auth needsOnboarding={true} />
        <FxLayer />
      </div>
    );
  }

  const hideNav = location.pathname.startsWith("/node/");

  return (
    <div className="app-frame">
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/map" element={<MapScreen />} />
          <Route path="/node/:code" element={<NodeScreen />} />
          <Route path="/teacher" element={<TeacherScreen />} />
          <Route path="/profile" element={<ProfileScreen />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      {!hideNav && (
        <nav className="tabbar" aria-label="Основная навигация">
          <NavLink to="/" end className={({ isActive }) => `tab ${isActive ? "active" : ""}`}>
            <span className="tab-glyph">⌂</span>
            <span className="tab-label">Главная</span>
          </NavLink>
          <NavLink to="/map" className={({ isActive }) => `tab ${isActive ? "active" : ""}`}>
            <span className="tab-glyph">✦</span>
            <span className="tab-label">Карта</span>
          </NavLink>
          <NavLink to="/teacher" className={({ isActive }) => `tab ${isActive ? "active" : ""}`}>
            <span className="tab-glyph">◉</span>
            <span className="tab-label">Учитель</span>
          </NavLink>
          <NavLink to="/profile" className={({ isActive }) => `tab ${isActive ? "active" : ""}`}>
            <span className="tab-glyph">⬡</span>
            <span className="tab-label">Профиль</span>
          </NavLink>
        </nav>
      )}
      <FxLayer />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <GameProvider>
        <Starfield />
        <div className="nebula nebula-a" />
        <div className="nebula nebula-b" />
        <Shell />
      </GameProvider>
    </BrowserRouter>
  );
}
