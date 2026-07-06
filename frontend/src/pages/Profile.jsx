import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import AppHeader from "../components/AppHeader";
import API_BASE from "../config";
import "../App.css";

function getStoredUser() {
  return localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
}

function setStoredUser(userObj) {
  if (localStorage.getItem("auraiUser")) {
    localStorage.setItem("auraiUser", JSON.stringify(userObj));
  } else {
    sessionStorage.setItem("auraiUser", JSON.stringify(userObj));
  }
}

export default function Profile() {
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [profile, setProfile] = useState(null);

  const [draftAddress, setDraftAddress] = useState("");
  const [draftAllergies, setDraftAllergies] = useState("");
  const [editAddress, setEditAddress] = useState(false);
  const [editAllergies, setEditAllergies] = useState(false);

  // Change password state
  const [showPwSection, setShowPwSection] = useState(false);
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [pwError, setPwError] = useState("");
  const [pwSuccess, setPwSuccess] = useState("");
  const [pwSaving, setPwSaving] = useState(false);

  useEffect(() => {
    const stored = getStoredUser();
    if (!stored) { navigate("/login", { replace: true }); return; }
    let user;
    try { user = JSON.parse(stored); } catch { navigate("/login", { replace: true }); return; }

    const load = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`${API_BASE}/me`, { params: { user_id: user.id } });
        setProfile(res.data);
        setDraftAddress(res.data.address || "");
        setDraftAllergies(res.data.allergies || "");
      } catch { setError("Failed to load profile."); }
      finally { setLoading(false); }
    };
    load();
  }, [navigate]);

  const isEditing = editAddress || editAllergies;

  const cancelEdits = () => {
    setDraftAddress(profile?.address || "");
    setDraftAllergies(profile?.allergies || "");
    setEditAddress(false);
    setEditAllergies(false);
    setError("");
  };

  const confirmEdits = async () => {
    const stored = getStoredUser();
    if (!stored) return navigate("/login", { replace: true });
    let user;
    try { user = JSON.parse(stored); } catch { return navigate("/login", { replace: true }); }
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await axios.put(`${API_BASE}/me`,
        { address: draftAddress.trim() || null, allergies: draftAllergies.trim() || null },
        { params: { user_id: user.id } }
      );
      const updated = res.data.user;
      setProfile(updated);
      setDraftAddress(updated.address || "");
      setDraftAllergies(updated.allergies || "");
      setEditAddress(false);
      setEditAllergies(false);
      setStoredUser({ ...user, name: updated.name, email: updated.email, role: updated.role || user.role || "user" });
      setSuccess("Saved!");
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to save changes.");
    } finally { setSaving(false); }
  };

  const handleChangePassword = async () => {
    setPwError(""); setPwSuccess("");
    if (!currentPw || !newPw || !confirmPw) { setPwError("All fields are required."); return; }
    if (newPw !== confirmPw) { setPwError("New passwords don't match."); return; }
    if (newPw.length < 8) { setPwError("New password must be at least 8 characters."); return; }
    const stored = getStoredUser();
    if (!stored) return navigate("/login", { replace: true });
    let user;
    try { user = JSON.parse(stored); } catch { return navigate("/login", { replace: true }); }
    setPwSaving(true);
    try {
      await axios.post(`${API_BASE}/me/change-password`,
        { current_password: currentPw, new_password: newPw },
        { params: { user_id: user.id } }
      );
      setPwSuccess("Password updated successfully.");
      setCurrentPw(""); setNewPw(""); setConfirmPw("");
      setShowPwSection(false);
    } catch (e) {
      setPwError(e?.response?.data?.detail || "Failed to change password.");
    } finally { setPwSaving(false); }
  };

  const initials = profile ? (profile.name || profile.email || "?").trim().slice(0, 2).toUpperCase() : "?";
  const isAdmin = String(profile?.role || "user").toLowerCase() === "admin";

  return (
    <div className="app-root">
      <AppHeader />
      <div style={{ padding: "20px 16px 80px", width: "100%", boxSizing: "border-box" }}>

        {loading && <p className="hint">Loading...</p>}
        {error && !isEditing && <p className="error-msg">{error}</p>}

        {!loading && profile && (
          <>
            {/* Profile header */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
              <div className="admin-user-avatar" style={{ width: 56, height: 56, fontSize: 20 }}>
                {initials}
              </div>
              <div>
                <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
                  {profile.name || "User"}
                </div>
                <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 2 }}>{profile.email}</div>
                <div style={{ marginTop: 6 }}>
                  <span className={`admin-role-badge ${isAdmin ? "admin-role-admin" : "admin-role-user"}`}>
                    {isAdmin ? "Admin" : "User"}
                  </span>
                </div>
              </div>
            </div>

            {/* Basic details card */}
            <div className="admin-user-card" style={{ flexDirection: "column", alignItems: "stretch", gap: 0, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 14, color: "var(--text-primary)" }}>Basic details</div>
              {[
                { label: "Name", value: profile.name },
                { label: "Email", value: profile.email },
                { label: "Phone", value: profile.phone },
                { label: "Age", value: profile.age },
              ].map(({ label, value }) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid rgba(170,151,241,0.12)" }}>
                  <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{label}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{value || "—"}</span>
                </div>
              ))}
            </div>

            {/* Address card */}
            <div className="admin-user-card" style={{ flexDirection: "column", alignItems: "stretch", gap: 0, marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)" }}>Address</span>
                {!editAddress && (
                  <button className="ghost-btn admin-open-btn" onClick={() => { setError(""); setEditAddress(true); }}>Edit</button>
                )}
              </div>
              <textarea
                className="profile-textarea"
                rows={3}
                value={draftAddress}
                onChange={(e) => setDraftAddress(e.target.value)}
                disabled={!editAddress}
                placeholder="Add your address..."
              />
            </div>

            {/* Allergies card */}
            <div className="admin-user-card" style={{ flexDirection: "column", alignItems: "stretch", gap: 0, marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <span style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)" }}>Allergies / sensitivities</span>
                {!editAllergies && (
                  <button className="ghost-btn admin-open-btn" onClick={() => { setError(""); setEditAllergies(true); }}>Edit</button>
                )}
              </div>
              <textarea
                className="profile-textarea"
                rows={3}
                value={draftAllergies}
                onChange={(e) => setDraftAllergies(e.target.value)}
                disabled={!editAllergies}
                placeholder="e.g. fragrance, niacinamide, nuts..."
              />
            </div>

            {/* Save / cancel when editing */}
            {isEditing && (
              <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                <button className="ghost-btn" style={{ flex: 1 }} onClick={cancelEdits} disabled={saving}>Cancel</button>
                <button className="primary-btn" style={{ flex: 1 }} onClick={confirmEdits} disabled={saving}>
                  {saving ? "Saving..." : "Save changes"}
                </button>
              </div>
            )}
            {success && <p style={{ color: "#22c55e", fontWeight: 700, marginBottom: 12, fontSize: 14 }}>{success}</p>}
            {error && isEditing && <p className="error-msg">{error}</p>}

            {/* Change password */}
            <div className="admin-user-card" style={{ flexDirection: "column", alignItems: "stretch", gap: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)" }}>Password</span>
                <button className="ghost-btn admin-open-btn" onClick={() => { setShowPwSection(!showPwSection); setPwError(""); setPwSuccess(""); }}>
                  {showPwSection ? "Cancel" : "Change"}
                </button>
              </div>

              {pwSuccess && <p style={{ color: "#22c55e", fontWeight: 700, marginTop: 10, fontSize: 14 }}>{pwSuccess}</p>}

              {showPwSection && (
                <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                  <input className="auth-input" type="password" placeholder="Current password" value={currentPw} onChange={(e) => setCurrentPw(e.target.value)} />
                  <input className="auth-input" type="password" placeholder="New password (min 8 chars)" value={newPw} onChange={(e) => setNewPw(e.target.value)} />
                  <input className="auth-input" type="password" placeholder="Confirm new password" value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)} />
                  {pwError && <p className="error-msg" style={{ margin: 0 }}>{pwError}</p>}
                  <button className="primary-btn" onClick={handleChangePassword} disabled={pwSaving}>
                    {pwSaving ? "Updating..." : "Update password"}
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
