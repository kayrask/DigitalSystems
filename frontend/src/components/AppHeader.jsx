import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import "../App.css";

function getStoredUser() {
  return localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
}

function safeParseUser(stored) {
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    // corrupted storage -> clear both
    localStorage.removeItem("auraiUser");
    sessionStorage.removeItem("auraiUser");
    return null;
  }
}

function AppHeader() {
  const navigate = useNavigate();

  const stored = getStoredUser();
  const user = safeParseUser(stored);

  const handleLogout = () => {
    localStorage.removeItem("auraiUser");
    sessionStorage.removeItem("auraiUser");
    navigate("/", { replace: true });
  };

  const displayName = user ? (user.name || user.email || "User") : "";
  const initial = user ? displayName.charAt(0).toUpperCase() : "";
  const isAdmin = String(user?.role || "user").toLowerCase() === "admin";

  const linkClass = ({ isActive }) =>
    "nav-link" + (isActive ? " nav-link-active" : "");

  return (
    <header className={"app-header" + (user ? " app-header-auth" : "")}>
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
          <NavLink to="/history" className={linkClass}>
            Recent scans
          </NavLink>
          <NavLink to="/scan" className={linkClass}>
            Get a new scan
          </NavLink>
          {isAdmin && (
            <NavLink to="/admin/users" className={linkClass}>
              Admin
            </NavLink>
          )}
        </nav>
      )}

      <div className="header-actions">
        {user ? (
          <>
            <button className="account-pill" onClick={() => navigate("/profile")}>
              <span className="account-initial">{initial}</span>
              <span className="account-name">{displayName}</span>
            </button>

            <button className="primary-btn logout-btn" onClick={handleLogout}>
              Log out
            </button>
          </>
        ) : (
          <>
            <button className="ghost-btn" onClick={() => navigate("/login")}>
              Log in
            </button>
            <button className="primary-btn" onClick={() => navigate("/register")}>
              Register
            </button>
          </>
        )}
      </div>
    </header>
  );
}

export default AppHeader;
