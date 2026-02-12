import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import AppHeader from "../components/AppHeader";
import "../App.css";

const API_BASE = "http://127.0.0.1:8000";

export default function AdminUsers() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [items, setItems] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
    if (!stored) {
      navigate("/login", { replace: true });
      return;
    }
    let user = null;
    try {
      user = JSON.parse(stored);
    } catch {
      navigate("/login", { replace: true });
      return;
    }
    if (String(user?.role || "user").toLowerCase() !== "admin") {
      navigate("/history", { replace: true });
      return;
    }

    const load = async () => {
      try {
        setLoading(true);
        const res = await axios.get(`${API_BASE}/admin/users`, {
          params: { user_id: user.id, limit: 300 },
        });
        setItems(res.data.items || []);
      } catch (e) {
        console.error(e);
        setError(e?.response?.data?.detail || "Failed to load users.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [navigate]);

  return (
    <div className="app-root">
      <AppHeader />
      <main className="app-main">
        <section className="card dashboard-card">
          <h2>Admin - Accounts</h2>
          <p className="hint">Open an account to view scans, formulas and annotations.</p>
          {loading && <p className="hint">Loading users...</p>}
          {error && <p className="error-msg">{error}</p>}
          {!loading && !error && items.length === 0 && <p className="hint">No users found.</p>}
          {!loading && !error && items.length > 0 && (
            <div className="history-list">
              {items.map((u) => (
                <div key={u.id} className="history-item">
                  <div className="history-header">
                    <span className="history-date">{u.name || "Unnamed user"}</span>
                    <span style={{ fontSize: 12, opacity: 0.7 }}>{u.email}</span>
                  </div>
                  <div className="history-results" style={{ marginBottom: 10 }}>
                    <div className="history-chip history-chip-present">
                      <span className="history-chip-name">Role</span>
                      <span className="history-chip-percent">{String(u.role || "user").toUpperCase()}</span>
                    </div>
                    <div className="history-chip history-chip-absent">
                      <span className="history-chip-name">Scans</span>
                      <span className="history-chip-percent">{u.scan_count || 0}</span>
                    </div>
                  </div>
                  <button
                    className="primary-btn"
                    onClick={() => navigate(`/admin/users/${u.id}/scans`)}
                  >
                    Open Account
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

