import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import AppHeader from "../components/AppHeader";
import "../App.css";

import API_BASE from "../config";

function getStoredUser() {
  return localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
}

function setStoredUser(userObj) {
  // preserve remember-me choice
  if (localStorage.getItem("auraiUser")) {
    localStorage.setItem("auraiUser", JSON.stringify(userObj));
  } else {
    sessionStorage.setItem("auraiUser", JSON.stringify(userObj));
  }
}

function Profile() {
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [profile, setProfile] = useState(null);

  // editable drafts
  const [draftAddress, setDraftAddress] = useState("");
  const [draftAllergies, setDraftAllergies] = useState("");

  // edit mode toggles
  const [editAddress, setEditAddress] = useState(false);
  const [editAllergies, setEditAllergies] = useState(false);

  // Load profile
  useEffect(() => {
    const stored = getStoredUser();
    if (!stored) {
      navigate("/login", { replace: true });
      return;
    }

    let user;
    try {
      user = JSON.parse(stored);
    } catch {
      navigate("/login", { replace: true });
      return;
    }

    const load = async () => {
      setLoading(true);
      setError("");
      setSuccess("");

      try {
        const res = await axios.get(`${API_BASE}/me`, {
          params: { user_id: user.id },
        });

        setProfile(res.data);
        setDraftAddress(res.data.address || "");
        setDraftAllergies(res.data.allergies || "");
      } catch (e) {
        console.error(e);
        setError("Failed to load profile.");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [navigate]);

  const isEditing = editAddress || editAllergies;

  const startEditAddress = () => {
    setSuccess("");
    setError("");
    setEditAddress(true);
  };

  const startEditAllergies = () => {
    setSuccess("");
    setError("");
    setEditAllergies(true);
  };

  const cancelEdits = () => {
    setSuccess("");
    setError("");
    setDraftAddress(profile?.address || "");
    setDraftAllergies(profile?.allergies || "");
    setEditAddress(false);
    setEditAllergies(false);
  };

  const confirmEdits = async () => {
    const stored = getStoredUser();
    if (!stored) return navigate("/login", { replace: true });

    let user;
    try {
      user = JSON.parse(stored);
    } catch {
      return navigate("/login", { replace: true });
    }

    setSaving(true);
    setError("");
    setSuccess("");

    try {
      const payload = {
        address: draftAddress.trim() ? draftAddress.trim() : null,
        allergies: draftAllergies.trim() ? draftAllergies.trim() : null,
      };

      const res = await axios.put(`${API_BASE}/me`, payload, {
        params: { user_id: user.id },
      });

      // backend returns: {"ok": true, "user": {...}}
      const updated = res.data.user;

      setProfile(updated);
      setDraftAddress(updated.address || "");
      setDraftAllergies(updated.allergies || "");

      setEditAddress(false);
      setEditAllergies(false);

      // update stored user (so header shows updated name/email if changed later)
      setStoredUser({
        ...user,
        name: updated.name,
        email: updated.email,
        role: updated.role || user.role || "user",
      });

      setSuccess("Saved!");
    } catch (e) {
      console.error(e);
      setError(e?.response?.data?.detail || "Failed to save changes.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="app-root">
      <AppHeader />

      <main className="profile-main">
        <section className="card profile-card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <div>
              <h2 style={{ marginBottom: 6 }}>Account details</h2>
              <div className="profile-subhint">
                Address & allergies are editable. Age is locked for consistency.
              </div>
            </div>
          </div>

          {loading && <p className="hint">Loading...</p>}
          {error && <p className="error-msg">{error}</p>}
          {success && <p style={{ color: "green", fontWeight: 800 }}>{success}</p>}

          {!loading && profile && (
            <div style={{ display: "grid", gap: 16, marginTop: 16 }}>
              {/* BASIC DETAILS (READ-ONLY) */}
              <div className="profile-section">
                <div style={{ fontWeight: 900, marginBottom: 12 }}>Basic details</div>

                <div className="profile-row">
                  <div className="profile-label">Name</div>
                  <div className="profile-value">{profile.name || "—"}</div>
                </div>

                <div className="profile-row">
                  <div className="profile-label">Email</div>
                  <div className="profile-value">{profile.email || "—"}</div>
                </div>

                <div className="profile-row">
                  <div className="profile-label">Phone</div>
                  <div className="profile-value">{profile.phone || "—"}</div>
                </div>

                <div className="profile-row">
                  <div className="profile-label">Age (locked)</div>
                  <div className="profile-value locked">{profile.age ?? "—"}</div>
                </div>
              </div>

              {/* ADDRESS (EDITABLE WITH EDIT BUTTON) */}
              <div className="profile-section">
                <div className="profile-edit-header">
                  <div className="profile-edit-title">Address</div>
                  {!editAddress ? (
                    <button className="secondary-btn" onClick={startEditAddress}>
                      Edit
                    </button>
                  ) : (
                    <span className="editing-badge">Editing…</span>
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

              {/* ALLERGIES (EDITABLE WITH EDIT BUTTON) */}
              <div className="profile-section">
                <div className="profile-edit-header">
                  <div className="profile-edit-title">Allergies / sensitivities</div>
                  {!editAllergies ? (
                    <button className="secondary-btn" onClick={startEditAllergies}>
                      Edit
                    </button>
                  ) : (
                    <span className="editing-badge">Editing…</span>
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

              {/* ACTIONS ONLY WHEN EDITING */}
              {isEditing && (
                <div className="profile-actions">
                  <button className="secondary-btn" onClick={cancelEdits} disabled={saving}>
                    Cancel
                  </button>
                  <button className="primary-btn" onClick={confirmEdits} disabled={saving}>
                    {saving ? "Saving..." : "Confirm changes"}
                  </button>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>⚠️ Demo only — not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default Profile;
