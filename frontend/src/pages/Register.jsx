// src/pages/Register.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";

import API_BASE from "../config";

function Register() {
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // optional
  const [address, setAddress] = useState("");

  // allergies flow
  const [hasAllergies, setHasAllergies] = useState(null); // null | "yes" | "no"
  const [allergies, setAllergies] = useState("");

  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");

    if (!name.trim() || !email.trim() || !password) {
      setErrorMsg("Please fill in all required fields.");
      return;
    }

    if (!email.includes("@") || !email.includes(".")) {
      setErrorMsg("Please enter a valid email address.");
      return;
    }

    if (password.length < 8) {
      setErrorMsg("Password must be at least 8 characters.");
      return;
    }

    // If user chose "yes" but left allergies empty, warn (optional rule)
    if (hasAllergies === "yes" && !allergies.trim()) {
      setErrorMsg("Please list your allergies or select 'No'.");
      return;
    }

    try {
      const response = await axios.post(`${API_BASE}/register`, {
        name: name.trim(),
        email: email.trim().toLowerCase(),
        password,
        address: address.trim() ? address.trim() : null, // optional
        allergies: hasAllergies === "yes" ? allergies.trim() : null,
      });

      // default: session login after register (matches your remember-me behavior)
      sessionStorage.setItem("auraiUser", JSON.stringify(response.data));
      localStorage.removeItem("auraiUser");

      setSuccessMsg("Account created! Redirecting…");
      setTimeout(() => navigate("/", { replace: true }), 900);
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setErrorMsg(err.response.data.detail);
      } else {
        setErrorMsg("Registration failed. Please try again.");
      }
    }
  };

  const selectAllergies = (choice) => {
    setHasAllergies(choice);
    setErrorMsg("");
    setSuccessMsg("");
    if (choice === "no") setAllergies(""); // clear if they say no
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
        <section className="card auth-card auth-shell">
          <aside className="auth-aside">
            <p className="auth-aside-kicker">Create your account</p>
            <h2>Set up your personal AURAI space</h2>
            <p className="auth-aside-copy">
              Your profile stores scan history and preferences to provide consistent, trackable results.
            </p>
            <div className="auth-aside-points">
              <span>Secure account-based history</span>
              <span>Faster repeat scan flow</span>
              <span>Personalized routine context</span>
            </div>
          </aside>

          <div className="auth-panel">
            <h3>Create an account</h3>
            <p className="hint">Registration takes less than a minute.</p>

            <form className="auth-form" onSubmit={handleSubmit}>
              <label className="input-label">
                Name *
                <input
                  type="text"
                  className="input-field"
                  placeholder="Your name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </label>

              <label className="input-label">
                Email *
                <input
                  type="email"
                  className="input-field"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </label>

              <label className="input-label">
                Password *
                <input
                  type="password"
                  className="input-field"
                  placeholder="Min. 8 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </label>

              <label className="input-label">
                Address (optional)
                <textarea
                  className="input-field"
                  placeholder="Street, city, postcode (optional)"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  rows={3}
                  style={{ resize: "vertical" }}
                />
              </label>

              <div className="input-label" style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>
                  Do you have any allergies?
                </div>

                <div className="auth-toggle-row">
                  <button
                    type="button"
                    className={hasAllergies === "yes" ? "mode-btn active" : "mode-btn"}
                    onClick={() => selectAllergies("yes")}
                  >
                    Yes
                  </button>

                  <button
                    type="button"
                    className={hasAllergies === "no" ? "mode-btn active" : "mode-btn"}
                    onClick={() => selectAllergies("no")}
                  >
                    No
                  </button>
                </div>

                {hasAllergies === "yes" && (
                  <textarea
                    className="input-field"
                    placeholder="List your allergies (e.g. fragrance, nuts, niacinamide...)"
                    value={allergies}
                    onChange={(e) => setAllergies(e.target.value)}
                    rows={3}
                    style={{ marginTop: 10, resize: "vertical" }}
                  />
                )}
              </div>

              <div className="auth-actions">
                <button type="submit" className="primary-btn">
                  Register
                </button>
                <Link to="/login" className="secondary-link">
                  Already have an account? Log in
                </Link>
              </div>

              {errorMsg && <p className="error-msg">{errorMsg}</p>}
              {successMsg && <p className="success-msg">{successMsg}</p>}
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

export default Register;
