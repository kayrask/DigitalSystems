// src/pages/Login.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";

const API_BASE = "http://127.0.0.1:8000"; // same as Register.jsx

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

    try {
      const response = await axios.post(`${API_BASE}/login`, {
        email,
        password,
      });

      const user = response.data;

      if (rememberMe) {
        localStorage.setItem("auraiUser", JSON.stringify(user));
        sessionStorage.removeItem("auraiUser");
      } 
        else {
        sessionStorage.setItem("auraiUser", JSON.stringify(user));
        localStorage.removeItem("auraiUser");
      }

      navigate("/", { replace: true });

    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setErrorMsg(err.response.data.detail);
      } else {
        setErrorMsg("Login failed. Please try again.");
      }
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand brand-left" onClick={() => navigate("/") }>
          <div className="brand-icon">🌬️</div>
          <div>
            <h1>AURAI</h1>
            <p>AI-powered facial skin analysis</p>
          </div>
        </div>
      </header>

      <main className="app-main auth-main">
        <section className="card auth-card">
          <h2>Log in</h2>
          <p className="hint">Log in with the account you registered.</p>

          <form className="auth-form" onSubmit={handleSubmit}>
            <label className="input-label">
              Email
              <input
                type="email"
                className="input-field"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>

            <label className="input-label">
              Password
              <input
                type="password"
                className="input-field"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              Remember me
            </label>


            <div className="auth-actions">
              <button type="submit" className="primary-btn">
                Log in
              </button>
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
