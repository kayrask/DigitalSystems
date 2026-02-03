// src/pages/Register.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";

const API_BASE = "http://127.0.0.1:8000";

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

    if (password.length < 6) {
      setErrorMsg("Password should be at least 6 characters.");
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
        <section className="card auth-card">
          <h2>Create an account</h2>
          <p className="hint">This uses a real SQL database on the backend.</p>

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
                placeholder="Choose a password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </label>

            {/* Optional Address */}
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

            {/* Allergies Yes/No */}
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
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Register;
