import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import AppHeader from "../components/AppHeader";
import API_BASE from "../config";
import "../App.css";

export default function AdminUsers() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [items, setItems] = useState([]);
  const [currentUserId, setCurrentUserId] = useState(null);

  // Grant admin panel
  const [grantEmail, setGrantEmail] = useState("");
  const [grantStatus, setGrantStatus] = useState(null); // { type: "success"|"error", msg }
  const [granting, setGranting] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
    if (!stored) { navigate("/login", { replace: true }); return; }
    let user = null;
    try { user = JSON.parse(stored); } catch { navigate("/login", { replace: true }); return; }
    if (String(user?.role || "user").toLowerCase() !== "admin") {
      navigate("/history", { replace: true }); return;
    }
    setCurrentUserId(user.id);
    const load = async () => {
      try {
        setLoading(true);
        const res = await axios.get(`${API_BASE}/admin/users`, {
          params: { user_id: user.id, limit: 300 },
        });
        setItems(res.data.items || []);
      } catch (e) {
        setError(e?.response?.data?.detail || "Failed to load users.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [navigate]);

  const handleGrantAdmin = async (e) => {
    e.preventDefault();
    const email = grantEmail.trim().toLowerCase();
    if (!email) return;

    const target = items.find(u => (u.email || "").toLowerCase() === email);
    if (!target) {
      setGrantStatus({ type: "error", msg: `No account found for ${email}` });
      return;
    }
    if (target.id === currentUserId) {
      setGrantStatus({ type: "error", msg: "You already have admin access." });
      return;
    }
    if (String(target.role || "user").toLowerCase() === "admin") {
      setGrantStatus({ type: "error", msg: `${target.name || email} is already an admin.` });
      return;
    }

    setGranting(true);
    setGrantStatus(null);
    try {
      await axios.patch(`${API_BASE}/admin/users/${target.id}/role`, {
        requester_id: currentUserId,
        role: "admin",
      });
      setItems(prev => prev.map(u => u.id === target.id ? { ...u, role: "admin" } : u));
      setGrantStatus({ type: "success", msg: `${target.name || email} is now an admin.` });
      setGrantEmail("");
    } catch (err) {
      setGrantStatus({ type: "error", msg: err?.response?.data?.detail || "Failed to grant access." });
    } finally {
      setGranting(false);
    }
  };

  return (
    <div className="app-root">
      <AppHeader />
      <div style={{ padding: "20px 16px 80px", width: "100%", boxSizing: "border-box" }}>
        <h2 className="page-title" style={{ margin: "0 0 4px" }}>Accounts</h2>
        <p className="page-subtitle" style={{ margin: "0 0 20px" }}>Tap an account to view scans and annotations.</p>

        {/* Grant admin panel */}
        <div className="admin-grant-panel">
          <div className="admin-grant-title">Grant admin access</div>
          <form className="admin-grant-form" onSubmit={handleGrantAdmin}>
            <input
              className="admin-grant-input"
              type="email"
              placeholder="Enter account email..."
              value={grantEmail}
              onChange={e => { setGrantEmail(e.target.value); setGrantStatus(null); }}
              disabled={granting}
            />
            <button className="admin-grant-btn" type="submit" disabled={granting || !grantEmail.trim()}>
              {granting ? "Granting…" : "Grant"}
            </button>
          </form>
          {grantStatus && (
            <p className={`admin-grant-status ${grantStatus.type === "success" ? "admin-grant-ok" : "admin-grant-err"}`}>
              {grantStatus.type === "success" ? "✓ " : "✕ "}{grantStatus.msg}
            </p>
          )}
        </div>

        {loading && <p className="hint">Loading users...</p>}
        {error && <p className="error-msg">{error}</p>}
        {!loading && !error && items.length === 0 && (
          <p className="hint">No users found.</p>
        )}

        {!loading && !error && items.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {items.map((u) => {
              const isAdmin = String(u.role || "user").toLowerCase() === "admin";
              const isSelf = u.id === currentUserId;
              const initials = (u.name || u.email || "?").trim().slice(0, 2).toUpperCase();
              return (
                <div key={u.id} className="admin-user-card" onClick={() => navigate(`/admin/users/${u.id}/scans`)}>
                  <div className="admin-user-avatar">{initials}</div>
                  <div className="admin-user-info">
                    <div className="admin-user-name">
                      {u.name || "Unnamed"}
                      {isSelf && <span className="admin-self-tag"> (you)</span>}
                    </div>
                    <div className="admin-user-email">{u.email}</div>
                    <div className="admin-user-meta">
                      <span className={`admin-role-badge ${isAdmin ? "admin-role-admin" : "admin-role-user"}`}>
                        {isAdmin ? "Admin" : "User"}
                      </span>
                      <span className="admin-scan-count">{u.scan_count || 0} scans</span>
                    </div>
                  </div>
                  <button
                    className="ghost-btn admin-open-btn"
                    onClick={(e) => { e.stopPropagation(); navigate(`/admin/users/${u.id}/scans`); }}
                  >
                    Open
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
