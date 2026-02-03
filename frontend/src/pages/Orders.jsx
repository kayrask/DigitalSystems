// src/pages/Orders.jsx
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";

function Orders() {
  const navigate = useNavigate();

  useEffect(() => {
    const stored =
      localStorage.getItem("auraiUser") ||
      sessionStorage.getItem("auraiUser");

    if (!stored) {
      navigate("/login", { replace: true });
    }
  }, [navigate]);

  return (
    <div className="app-root">
      <AppHeader />

      <main className="app-main">
        <section className="card dashboard-card">
          <h2>Your orders</h2>
          <p className="muted">
            This is a placeholder. Once payment is integrated, your purchased
            scans and receipts will appear here.
          </p>
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Orders;
