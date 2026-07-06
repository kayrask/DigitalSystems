// src/pages/Login.jsx
import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";
import API_BASE from "../config";
import AppHeader from "../components/AppHeader";

function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [rememberMe, setRememberMe] = useState(false);

  useEffect(() => {
    function tryInit() {
      if (typeof window.initLiquidEther === 'function') {
        window.initLiquidEther('liquid-hero-bg');
      } else {
        setTimeout(tryInit, 50);
      }
    }
    tryInit();
    return () => { if (window._leDestroy) window._leDestroy(); };
  }, []);

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
    <div className="app-root auth-page">
      <div id="liquid-hero-bg" className="home-hero-fluid" />
      <AppHeader />

      <main className="auth-wrap">
        <section className="auth-card-v2">
          <div className="auth-card-head">
            <p className="home-hero-kicker">Welcome back</p>
            <h2 className="auth-title-v2">
              Log in to <span className="home-hero-gradient">NYMIRA</span>
            </h2>
            <p className="auth-sub-v2">Pick up right where your skin left off.</p>
          </div>

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
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Login;
