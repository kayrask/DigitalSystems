import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";
const UI_DETECT_THRESHOLD = 0.5;

const prettyName = {
  acne: "Acne",
  bags: "Under-eye bags",
  blackheads: "Blackheads",
  hyperpigmentation: "Hyperpigmentation",
  redness: "Redness",
};

const driverToFriendly = (d) => {
  if (!d) return null;
  const s = String(d).toLowerCase();

  // benefit drivers
  if (s.includes("contains salicylic") || s.includes("contains bha"))
    return "Includes BHA exfoliation to help unclog pores.";
  if (s.includes("contains glycolic") || s.includes("contains aha") || s.includes("contains lactic"))
    return "Includes gentle exfoliation to improve texture and tone.";
  if (s.includes("contains niacinamide"))
    return "Includes niacinamide to support the skin barrier and reduce unevenness.";
  if (s.includes("contains panthenol") || s.includes("contains allantoin") || s.includes("contains centella"))
    return "Includes soothing ingredients to help calm visible redness.";

  // risk drivers
  if (s.includes("irritant flag: fragrance") || s.includes("irritant flag: parfum") || s.includes("irritant flag: aroma"))
    return "Contains fragrance, which may irritate sensitive skin.";
  if (s.includes("strong active: retinol") || s.includes("strong active: retinal"))
    return "Contains retinoids, which can increase irritation/dryness at first.";
  if (s.includes("strong active: salicylic") || s.includes("strong active: bha"))
    return "Contains exfoliating acids (BHA) that may increase irritation if overused.";
  if (s.includes("strong active: lactic") || s.includes("strong active: glycolic") || s.includes("strong active: aha"))
    return "Contains exfoliating acids (AHA) that may increase irritation if overused.";
  if (s.includes("strong active: vitamin c") || s.includes("strong active: ascorb"))
    return "Contains vitamin C, which may irritate if your skin is sensitive.";
  if (s.includes("multiple strong actives"))
    return "Multiple strong actives detected — introduce slowly and patch test.";
  if (s.includes("skin type: sensitive"))
    return "Sensitive skin type detected — irritation risk is higher.";

  // fallback
  return String(d);
};

const formatDrivers = (drivers = []) =>
  (drivers || [])
    .map(driverToFriendly)
    .filter(Boolean)
    .slice(0, 4);

function History() {
  const navigate = useNavigate();
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState(null);

  // Phase 4A: explainability state (shared but reset per open scan)
  const [explainTarget, setExplainTarget] = useState(null);
  const [explainOverlay, setExplainOverlay] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [explainCache, setExplainCache] = useState({}); // `${scanId}:${target}` -> base64

  useEffect(() => {
    const stored =
      localStorage.getItem("auraiUser") ||
      sessionStorage.getItem("auraiUser");

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

    const fetchScans = async () => {
      try {
        const res = await axios.get(`${API_BASE}/scans`, {
          params: { user_id: user.id, limit: 20 },
        });
        setScans(res.data.items || []);
      } catch (err) {
        console.error("Failed to load scans:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchScans();
  }, [navigate]);

  const toggle = (id) => {
    setOpenId((prev) => {
      const next = prev === id ? null : id;

      // reset explain UI when changing which scan is open
      setExplainTarget(null);
      setExplainOverlay(null);
      setExplainLoading(false);
      setExplainError("");

      return next;
    });
  };

  const fetchExplain = async (scanId, target) => {
    const key = `${scanId}:${target}`;
    setExplainError("");
    setExplainTarget(target);

    // cached?
    if (explainCache[key]) {
      setExplainOverlay(explainCache[key]);
      return;
    }

    setExplainLoading(true);
    setExplainOverlay(null);

    try {
      const res = await axios.get(`${API_BASE}/explain`, {
        params: { scan_id: scanId, target },
      });

      const b64 = res.data?.overlay_png_base64;
      if (!b64) {
        setExplainError("No overlay returned.");
        return;
      }

      setExplainOverlay(b64);
      setExplainCache((prev) => ({ ...prev, [key]: b64 }));
    } catch (err) {
      console.error("Explain failed:", err);
      const msg =
        err?.response?.data?.detail ||
        "Failed to generate overlay (scan image may not be stored yet).";
      setExplainError(msg);
    } finally {
      setExplainLoading(false);
    }
  };

  // Handles both OLD and NEW scan formats
  const unpack = (scan) => {
    if (!scan?.results) return {};

    // NEW format (nested)
    if (scan.results.results) {
      return {
        concerns: scan.results.results,
        skinType: scan.results.skin_type || null,
        routine: scan.results.routine || null,
        outcome: scan.results.outcome || null,
        risk: scan.results.risk || null,
      };
    }

    // OLD format (only concerns dict)
    return {
      concerns: scan.results,
      skinType: null,
      routine: null,
      outcome: null,
      risk: null,
    };
  };

  const RiskBenefitWidget = ({ outcome, risk }) => {
    if (!outcome && !risk) return null;

    const benefit = outcome?.benefit || null;
    const riskLevel = risk?.level;
    const riskScore = risk?.irritation_score ?? null;

    const riskPillBg =
      riskLevel === "high"
        ? "rgba(220, 53, 69, 0.12)"
        : riskLevel === "medium"
        ? "rgba(255, 193, 7, 0.16)"
        : "rgba(40, 167, 69, 0.12)";

    return (
      <div
        style={{
          marginTop: 12,
          padding: 12,
          borderRadius: 12,
          background: "white",
          border: "1px solid rgba(0,0,0,0.06)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
          <h4 style={{ margin: 0 }}>Expected outcome & irritation risk</h4>

          {riskLevel && (
            <span
              style={{
                fontSize: 12,
                padding: "4px 10px",
                borderRadius: 999,
                background: riskPillBg,
                fontWeight: 900,
                whiteSpace: "nowrap",
              }}
            >
              Risk: {String(riskLevel).toUpperCase()}
              {typeof riskScore === "number" ? ` (${riskScore}%)` : ""}
            </span>
          )}
        </div>

        {benefit && (
          <div style={{ marginTop: 10 }}>
            {Object.entries(benefit).map(([k, v]) => {
              const score = v?.score ?? 0;
              const drivers = formatDrivers(v?.drivers || []);

              return (
                <div key={k} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                    <b>{prettyName[k] || k}</b>
                    <span style={{ opacity: 0.75 }}>{score}%</span>
                  </div>

                  <div style={{ height: 8, borderRadius: 999, background: "#eee", overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${Math.max(0, Math.min(score, 100))}%`,
                        height: "100%",
                        borderRadius: 999,
                        background: "linear-gradient(120deg, var(--accent), #c0a2ff)",
                      }}
                    />
                  </div>

                  {drivers.length > 0 && (
                    <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 12, opacity: 0.85 }}>
                      {drivers.map((line, i) => (
                        <li key={i}>{line}</li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {risk?.drivers?.length > 0 && (
          <div style={{ marginTop: 6 }}>
            <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 6 }}>
              Why this risk?
            </div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, opacity: 0.85 }}>
              {formatDrivers(risk.drivers).map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        )}

        <p style={{ fontSize: 11, opacity: 0.65, marginTop: 10, marginBottom: 0 }}>
          Proxy estimates for decision support — not a medical prediction.
        </p>
      </div>
    );
  };

  const ExplainabilityCard = ({ scanId }) => {
    return (
      <div
        style={{
          background: "white",
          padding: 12,
          borderRadius: 12,
          border: "1px solid rgba(0,0,0,0.06)",
          marginBottom: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
          <h4 style={{ margin: 0 }}>Explanation overlay </h4>
          <span style={{ fontSize: 11, opacity: 0.65 }}>
            Highlights areas influencing the prediction
          </span>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
          {["acne", "redness", "blackheads", "bags", "hyperpigmentation"].map((k) => {
            const active = explainTarget === k;
            const disabled = explainLoading && active;

            return (
              <button
                key={k}
                type="button"
                onClick={() => fetchExplain(scanId, k)}
                disabled={disabled}
                style={{
                  padding: "8px 10px",
                  fontSize: 12,
                  borderRadius: 999,
                  border: 0,
                  cursor: disabled ? "default" : "pointer",
                  background: active
                    ? "linear-gradient(120deg, #a98bff, #c0a2ff)"
                    : "#f1ecff",
                  color: active ? "#fff" : "var(--text-muted)",
                  opacity: disabled ? 0.7 : 1,
                }}
              >
                {prettyName[k] || k}
              </button>
            );
          })}
        </div>

        {explainLoading && (
          <p className="hint" style={{ marginTop: 10 }}>
            Generating overlay…
          </p>
        )}

        {explainError && (
          <p className="error-msg" style={{ marginTop: 10 }}>
            {explainError}
          </p>
        )}

        {explainOverlay && (
          <div style={{ marginTop: 12 }}>
            <img
              src={`data:image/png;base64,${explainOverlay}`}
              alt="Explainability overlay"
              style={{
                width: "100%",
                borderRadius: 12,
                border: "1px solid rgba(0,0,0,0.08)",
                display: "block",
              }}
            />
            <p style={{ fontSize: 11, opacity: 0.65, marginTop: 8, marginBottom: 0 }}>
              Warmer (redder) regions contributed more to the selected prediction.
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-root">
      <AppHeader />

      <main className="app-main">
        <section className="card dashboard-card">
          <h2>Recent scans</h2>

          {loading && <p className="hint">Loading your scan history…</p>}

          {!loading && scans.length === 0 && (
            <p className="hint">You don&apos;t have any saved scans yet.</p>
          )}

          {!loading && scans.length > 0 && (
            <div className="history-list">
              {scans.map((scan) => {
                const { concerns, skinType, routine, outcome, risk } = unpack(scan);
                const isOpen = openId === scan.id;

                return (
                  <div key={scan.id}>
                    {/* MAIN SCAN ROW */}
                    <div
                      className="history-item"
                      style={{
                        cursor: "pointer",
                        transition: "transform 0.15s ease, box-shadow 0.15s ease",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = "translateY(-1px)";
                        e.currentTarget.style.boxShadow = "0 10px 24px rgba(0,0,0,0.07)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = "translateY(0px)";
                        e.currentTarget.style.boxShadow = "none";
                      }}
                      onClick={() => toggle(scan.id)}
                    >
                      <div className="history-header">
                        <span className="history-date">
                          {new Date(scan.created_at).toLocaleString()}
                        </span>
                        <span style={{ fontSize: 12, opacity: 0.6 }}>
                          {isOpen ? "Hide details" : "View details"}
                        </span>
                      </div>

                      {skinType && (
                        <div style={{ fontSize: 13, marginBottom: 6 }}>
                          Skin type: <b>{skinType.skin_type?.toUpperCase()}</b>{" "}
                          ({((skinType.confidence || 0) * 100).toFixed(1)}%)
                        </div>
                      )}

                      <div className="history-results">
                        {concerns &&
                          Object.entries(concerns).map(([key, value]) => {
                            const prob = value?.probability || 0;
                            const present = prob >= UI_DETECT_THRESHOLD;
                            const percent = (prob * 100).toFixed(1);

                            return (
                              <div
                                key={key}
                                className={
                                  "history-chip " +
                                  (present ? "history-chip-present" : "history-chip-absent")
                                }
                              >
                                <span className="history-chip-name">
                                  {prettyName[key] || key}
                                </span>
                                <span className="history-chip-percent">{percent}%</span>
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    {/* EXPANDED DETAILS PANEL */}
                    <div
                      style={{
                        maxHeight: isOpen ? 1600 : 0,
                        overflow: "hidden",
                        transition: "max-height 0.4s ease",
                      }}
                    >
                      {isOpen && (
                        <div
                          style={{
                            marginTop: 10,
                            padding: 14,
                            borderRadius: 12,
                            background: "#f7f7f7",
                            maxHeight: 520,
                            overflowY: "auto",
                            WebkitOverflowScrolling: "touch",
                          }}
                        >
                          {/* Phase 4A: Explainability */}
                          <ExplainabilityCard scanId={scan.id} />

                          {/* Phase 3: Risk/Benefit */}
                          <RiskBenefitWidget outcome={outcome} risk={risk} />

                          {/* Routine / formulas */}
                          {routine && (
                            <>
                              <h4 style={{ marginTop: 12 }}>Generated routine & formulas</h4>
                              <p style={{ marginTop: -6, fontSize: 12, opacity: 0.65 }}>
                                Scroll inside this box to view full formula ↓
                              </p>

                              {routine.steps &&
                                routine.steps.map((step) => (
                                  <div
                                    key={step.step}
                                    style={{
                                      marginBottom: 12,
                                      background: "white",
                                      padding: 12,
                                      borderRadius: 10,
                                    }}
                                  >
                                    <div style={{ fontWeight: 900 }}>
                                      {String(step.step || "").toUpperCase()}
                                    </div>

                                    {step.changes?.length > 0 && (
                                      <ul style={{ fontSize: 13, marginTop: 6 }}>
                                        {step.changes.map((c, i) => (
                                          <li key={i}>
                                            {c.inci}: {c.from}% → {c.to}%
                                          </li>
                                        ))}
                                      </ul>
                                    )}

                                    {step.formula_personalized && (
                                      <table style={{ width: "100%", fontSize: 13, marginTop: 6 }}>
                                        <thead>
                                          <tr>
                                            <th align="left">INCI</th>
                                            <th align="right">%</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {step.formula_personalized.map((row, i) => (
                                            <tr key={i}>
                                              <td>{row.inci}</td>
                                              <td align="right">{row.percent}</td>
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    )}
                                  </div>
                                ))}
                            </>
                          )}

                          {!routine && !outcome && !risk && (
                            <p className="hint" style={{ marginTop: 0 }}>
                              No routine/risk data stored for this scan (older entry).
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
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

export default History;
