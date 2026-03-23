// src/pages/Login.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";
import API_BASE from "../config";

function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    if (!email.trim() || !password) {
      setErrorMsg("Please fill in both email and password.");
      return;
    }
    if (!email.includes("@") || !email.includes(".")) {
      setErrorMsg("Please enter a valid email address.");
      return;
    }

    try {
      const response = await axios.post(`${API_BASE}/login`, { email, password });
      const user = response.data;

      if (rememberMe) {
        localStorage.setItem("auraiUser", JSON.stringify(user));
        sessionStorage.removeItem("auraiUser");
      } else {
        sessionStorage.setItem("auraiUser", JSON.stringify(user));
        localStorage.removeItem("auraiUser");
      }

      navigate("/", { replace: true });
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || "Login failed. Please try again.");
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand brand-left" onClick={() => navigate("/")}>
          <div className="brand-icon">🌬️</div>
          <div>
            <h1>AURAI</h1>
            <p>AI-powered facial skin analysis</p>
          </div>
        </div>
      </header>

      <main className="app-main auth-main">
        <section className="card auth-card auth-shell">
          <aside className="auth-aside">
            <p className="auth-aside-kicker">Welcome back</p>
            <h2>Sign in to continue your skin journey</h2>
            <p className="auth-aside-copy">
              Access your scan history, monitor trends, and continue with a new analysis.
            </p>
            <div className="auth-aside-points">
              <span>Fast face scan workflow</span>
              <span>Structured concern tracking</span>
              <span>Private account-level history</span>
            </div>
          </aside>

          <div className="auth-panel">
            <h3>Log in</h3>
            <p className="hint">Use the account you registered previously.</p>

            <form className="auth-form" onSubmit={handleSubmit}>
              <label className="input-label">
                Email
                <input type="email" className="input-field" placeholder="you@example.com"
                  value={email} onChange={(e) => setEmail(e.target.value)} />
              </label>

              <label className="input-label">
                Password
                <input type="password" className="input-field" placeholder="••••••••"
                  value={password} onChange={(e) => setPassword(e.target.value)} />
              </label>

              <label className="auth-remember">
                <input type="checkbox" checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)} />
                <span>Remember me on this device</span>
              </label>

              <div className="auth-actions">
                <button type="submit" className="primary-btn">Log in</button>
                <Link to="/register" className="secondary-link">
                  Don&apos;t have an account? Register
                </Link>
              </div>

              {errorMsg && <p className="error-msg">{errorMsg}</p>}
            </form>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Login;
