import React, { useEffect, useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import "../App.css";

function getStoredUser() {
  return localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
}

function safeParseUser(stored) {
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    localStorage.removeItem("auraiUser");
    sessionStorage.removeItem("auraiUser");
    return null;
  }
}

function AppHeader() {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === "/";
  const [overHero, setOverHero] = useState(isHome);

  useEffect(() => {
    if (!isHome) { setOverHero(false); return; }
    setOverHero(window.scrollY < window.innerHeight * 0.8);
    const onScroll = () => setOverHero(window.scrollY < window.innerHeight * 0.8);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [isHome]);

  const onHero = isHome && overHero;

  const stored = getStoredUser();
  const user = safeParseUser(stored);

  const handleLogout = () => {
    localStorage.removeItem("auraiUser");
    sessionStorage.removeItem("auraiUser");
    navigate("/", { replace: true });
  };

  const displayName = user ? (user.name || user.email || "User") : "";
  const isAdmin = String(user?.role || "user").toLowerCase() === "admin";

  const linkClass = ({ isActive }) =>
    "nav-link" + (isActive ? " nav-link-active" : "");

  const bottomLinkClass = ({ isActive }) =>
    "bottom-nav-item" + (isActive ? " bottom-nav-active" : "");

  return (
    <>
      <header className={"app-header" + (user ? " app-header-auth" : "") + (onHero ? " app-header-hero" : "")}>
        <div className="brand brand-left" onClick={() => navigate("/")}>
          <div className="brand-icon">🌬️</div>
          <div>
            <h1>AURAI</h1>
            <p>AI-powered facial skin analysis</p>
          </div>
        </div>

        {!user && <div className="header-spacer" />}

        {user && (
          <nav className="main-nav">
            <NavLink to="/history" className={linkClass}>Recent scans</NavLink>
            <NavLink to="/scan" className={linkClass}>New scan</NavLink>
            <NavLink to="/routine" className={linkClass}>Routine</NavLink>
            {isAdmin && (
              <NavLink to="/admin/users" className={linkClass}>Admin</NavLink>
            )}
          </nav>
        )}

        <div className="header-actions">
          {user ? (
            <>
              <button className="account-pill" onClick={() => navigate("/profile")}>
                <span className="account-initial">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="8" r="4"/>
                    <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                  </svg>
                </span>
                <span className="account-name">{displayName}</span>
              </button>
              <button className="primary-btn logout-btn" onClick={handleLogout}>
                Log out
              </button>
            </>
          ) : (
            <>
              <button className="ghost-btn" onClick={() => navigate("/login")}>Log in</button>
              <button className="primary-btn" onClick={() => navigate("/register")}>Register</button>
            </>
          )}
        </div>
      </header>

      {/* Bottom nav — only shown on mobile when logged in */}
      {user && (
        <nav className="bottom-nav">
          <NavLink to="/" end className={bottomLinkClass}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
              <polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            <span>Home</span>
          </NavLink>

          <NavLink to="/scan" className={bottomLinkClass}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/>
              <path d="M21 21l-4.35-4.35"/>
              <line x1="11" y1="8" x2="11" y2="14"/>
              <line x1="8" y1="11" x2="14" y2="11"/>
            </svg>
            <span>New scan</span>
          </NavLink>

          <NavLink to="/history" className={bottomLinkClass}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
            <span>History</span>
          </NavLink>

          <NavLink to="/routine" className={bottomLinkClass}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 11l3 3L22 4"/>
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
            </svg>
            <span>Routine</span>
          </NavLink>

          <NavLink to="/profile" className={bottomLinkClass}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4"/>
              <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
            </svg>
            <span>Profile</span>
          </NavLink>

          {isAdmin && (
            <NavLink to="/admin/users" className={bottomLinkClass}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
              </svg>
              <span>Admin</span>
            </NavLink>
          )}
        </nav>
      )}
    </>
  );
}

export default AppHeader;
