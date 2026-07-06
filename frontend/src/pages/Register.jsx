// src/pages/Register.jsx
import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../App.css";
import API_BASE from "../config";
import AppHeader from "../components/AppHeader";

function Register() {
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [address, setAddress] = useState("");
  const [hasAllergies, setHasAllergies] = useState(null);
  const [allergies, setAllergies] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

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
    if (hasAllergies === "yes" && !allergies.trim()) {
      setErrorMsg("Please list your allergies or select 'No'.");
      return;
    }

    try {
      const response = await axios.post(`${API_BASE}/register`, {
        name: name.trim(),
        email: email.trim().toLowerCase(),
        password,
        address: address.trim() ? address.trim() : null,
        allergies: hasAllergies === "yes" ? allergies.trim() : null,
      });

      sessionStorage.setItem("auraiUser", JSON.stringify(response.data));
      localStorage.removeItem("auraiUser");

      setSuccessMsg("Account created! Redirecting…");
      setTimeout(() => navigate("/", { replace: true }), 900);
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || "Registration failed. Please try again.");
    }
  };

  const selectAllergies = (choice) => {
    setHasAllergies(choice);
    setErrorMsg("");
    setSuccessMsg("");
    if (choice === "no") setAllergies("");
  };

  return (
    <div className="app-root auth-page">
      <div id="liquid-hero-bg" className="home-hero-fluid" />
      <AppHeader />

      <main className="auth-wrap">
        <section className="auth-card-v2 auth-card-wide">
          <div className="auth-card-head">
            <p className="home-hero-kicker">Create your account</p>
            <h2 className="auth-title-v2">
              Your skin journey <span className="home-hero-gradient">starts here.</span>
            </h2>
            <p className="auth-sub-v2">Registration takes less than a minute.</p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
              <label className="input-label">
                Name *
                <input type="text" className="input-field" placeholder="Your name"
                  value={name} onChange={(e) => setName(e.target.value)} required />
              </label>

              <label className="input-label">
                Email *
                <input type="email" className="input-field" placeholder="you@example.com"
                  value={email} onChange={(e) => setEmail(e.target.value)} required />
              </label>

              <label className="input-label">
                Password *
                <input type="password" className="input-field" placeholder="Min. 8 characters"
                  value={password} onChange={(e) => setPassword(e.target.value)} required />
              </label>

              <label className="input-label">
                Address (optional)
                <textarea className="input-field" placeholder="Street, city, postcode (optional)"
                  value={address} onChange={(e) => setAddress(e.target.value)}
                  rows={3} style={{ resize: "vertical" }} />
              </label>

              <div className="input-label" style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Do you have any allergies?</div>
                <div className="auth-toggle-row">
                  <button type="button"
                    className={hasAllergies === "yes" ? "mode-btn active" : "mode-btn"}
                    onClick={() => selectAllergies("yes")}>Yes</button>
                  <button type="button"
                    className={hasAllergies === "no" ? "mode-btn active" : "mode-btn"}
                    onClick={() => selectAllergies("no")}>No</button>
                </div>
                {hasAllergies === "yes" && (
                  <textarea className="input-field"
                    placeholder="List your allergies (e.g. fragrance, nuts, niacinamide...)"
                    value={allergies} onChange={(e) => setAllergies(e.target.value)}
                    rows={3} style={{ marginTop: 10, resize: "vertical" }} />
                )}
              </div>

              <div className="auth-actions">
                <button type="submit" className="primary-btn">Register</button>
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
