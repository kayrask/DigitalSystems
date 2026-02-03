// src/pages/Dashboard.jsx
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";

function Account() {
  const navigate = useNavigate();
  
  const [user, setUser] = useState(null);

  useEffect(() => {
    const stored =
        localStorage.getItem("auraiUser") ||
        sessionStorage.getItem("auraiUser");

    if (!stored) {
      navigate("/login", { replace: true });
      return;
    }
    try {
      setUser(JSON.parse(stored));
    } catch {
      localStorage.removeItem("auraiUser");
      sessionStorage.removeItem("auraiUser");
      navigate("/login", { replace: true });
    }
  }, [navigate]);

  if (!user) {
    return null;
  }

  return (
    <div className="app-root">
      <AppHeader />

      <main className="app-main">
        <section className="card dashboard-card">
          <h2>Welcome back, {user.name || "AURAI user"} 👋</h2>
          <p className="hint">
            From here you can start new face scans, review your previous results,
            and see your orders once payments are integrated.
          </p>

          <div className="dashboard-actions">
            <button
              className="primary-btn"
              onClick={() => navigate("/scan")}
            >
              Start a new face scan
            </button>
          </div>

          <div className="dashboard-grid">
            <div className="dashboard-section">
              <h3>Your recent scans</h3>
              <p className="muted">
                We&apos;ll show your last face scans here (date, device, main
                conditions detected). For now this is a placeholder.
              </p>
            </div>

            <div className="dashboard-section">
              <h3>Orders & payments</h3>
              <p className="muted">
                Once payment is implemented, you&apos;ll see your completed
                payments, receipts and active plans here.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Account;
