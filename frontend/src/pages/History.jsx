import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";
import axios from "axios";

import API_BASE from "../config";
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

const resolveImageUrl = (path) => {
  if (!path) return null;
  if (String(path).startsWith("http://") || String(path).startsWith("https://")) return path;
  if (String(path).startsWith("/")) return `${API_BASE}${path}`;
  return `${API_BASE}/${path}`;
};

function History({ adminMode = false, userIdOverride = null, title = "Recent scans" }) {
  const navigate = useNavigate();
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState(null);
  const [targetUser, setTargetUser] = useState(null);
  const [viewerUser, setViewerUser] = useState(null);

  // Phase 4A: explainability state (shared but reset per open scan)
  const [explainTarget, setExplainTarget] = useState(null);
  const [explainOverlay, setExplainOverlay] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [explainCache, setExplainCache] = useState({}); // `${scanId}:${target}` -> base64

  // Outcome simulation state (per scan, cached)
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState("");
  const [simCache, setSimCache] = useState({}); // scanId -> response
  const [simOpenForScan, setSimOpenForScan] = useState(null);

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
      setViewerUser(user);
    } catch {
      navigate("/login", { replace: true });
      return;
    }
    if (adminMode && String(user.role || "user").toLowerCase() !== "admin") {
      navigate("/history", { replace: true });
      return;
    }

    const fetchScans = async () => {
      try {
        if (adminMode && userIdOverride) {
          const res = await axios.get(`${API_BASE}/admin/users/${userIdOverride}/scans`, {
            params: { user_id: user.id, limit: 30 },
          });
          setTargetUser(res.data.target_user || null);
          setScans(res.data.items || []);
        } else {
          const res = await axios.get(`${API_BASE}/scans`, {
            params: { user_id: user.id, limit: 20 },
          });
          setScans(res.data.items || []);
        }
      } catch (err) {
        console.error("Failed to load scans:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchScans();
  }, [navigate, adminMode, userIdOverride]);

  const toggle = (id) => {
    setOpenId((prev) => {
      const next = prev === id ? null : id;

      // reset explain UI when changing which scan is open
      setExplainTarget(null);
      setExplainOverlay(null);
      setExplainLoading(false);
      setExplainError("");
      setSimError("");
      setSimLoading(false);
      setSimOpenForScan(null);

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

      if (res.data?.type === "overlay") {
        const b64 = res.data?.overlay_png_base64;
        if (!b64) {
          setExplainError("No overlay returned.");
          return;
        }
        setExplainOverlay(b64);
        setExplainCache((prev) => ({ ...prev, [key]: b64 }));
      } else if (res.data?.type === "boxes") {
        // Store detections data for rendering
        const detectionData = {
          type: "boxes",
          source: res.data.source,
          image_png_base64: res.data.image_png_base64,
          detections: res.data.detections || [],
          detection_count: res.data.detection_count || res.data.detections?.length || 0,
          face_size: res.data.face_size,
          full_size: res.data.full_size,
        };
        setExplainOverlay(detectionData);
        setExplainCache((prev) => ({ ...prev, [key]: detectionData }));
      } else {
        setExplainError("Unknown explain response type.");
      }
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

  const handleSignalExplain = (e, scanId, target) => {
    e.stopPropagation();
    if (openId === scanId && explainTarget === target) {
      setExplainTarget(null);
      setExplainOverlay(null);
      setExplainError("");
      setExplainLoading(false);
      return;
    }
    if (openId !== scanId) {
      setOpenId(scanId);
      setExplainOverlay(null);
      setExplainError("");
    }
    fetchExplain(scanId, target);
  };

  const fetchOutcomeSimulation = async (scanId) => {
    if (!viewerUser?.id) return;
    setSimError("");
    setSimOpenForScan(scanId);

    setSimLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/simulate_outcome`, {
        params: {
          scan_id: scanId,
          user_id: viewerUser.id,
          strength: 1.0,
          t: Date.now(), // bust client/proxy cache while tuning simulation
        },
      });
      if (!res.data?.expected_png_base64 || !res.data?.current_png_base64) {
        setSimError("No outcome simulation returned.");
        return;
      }
      setSimCache((prev) => ({ ...prev, [scanId]: res.data }));
    } catch (err) {
      console.error("Outcome simulation failed:", err);
      setSimError(
        err?.response?.data?.detail || "Failed to generate expected outcome preview."
      );
    } finally {
      setSimLoading(false);
    }
  };

  const handleOutcomeToggle = async (scanId) => {
    if (simOpenForScan === scanId) {
      setSimOpenForScan(null);
      setSimError("");
      return;
    }
    await fetchOutcomeSimulation(scanId);
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

    const riskLevelClass =
      riskLevel === "high"
        ? " risk-high"
        : riskLevel === "medium"
        ? " risk-medium"
        : " risk-low";

    return (
      <div className="history-widget risk-widget">
        <div className="history-widget-head">
          <h4>Expected outcome & irritation risk</h4>

          {riskLevel && (
            <span className={"risk-pill" + riskLevelClass}>
              Risk: {String(riskLevel).toUpperCase()}
              {typeof riskScore === "number" ? ` (${riskScore}%)` : ""}
            </span>
          )}
        </div>

        {benefit && (
          <div className="benefit-list">
            {Object.entries(benefit).map(([k, v]) => {
              const score = v?.score ?? 0;
              const drivers = formatDrivers(v?.drivers || []);

              return (
                <div key={k} className="benefit-item">
                  <div className="benefit-row">
                    <b>{prettyName[k] || k}</b>
                    <span>{score}%</span>
                  </div>

                  <div className="benefit-bar">
                    <div
                      className="benefit-bar-fill"
                      style={{
                        width: `${Math.max(0, Math.min(score, 100))}%`,
                      }}
                    />
                  </div>

                  {drivers.length > 0 && (
                    <ul className="benefit-driver-list">
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
          <div className="risk-driver-block">
            <div className="risk-driver-title">
              Why this risk?
            </div>
            <ul className="risk-driver-list">
              {formatDrivers(risk.drivers).map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        )}

        <p className="risk-footnote">
          Proxy estimates for decision support — not a medical prediction.
        </p>
      </div>
    );
  };

  const ExplainabilityCard = ({ scanId }) => {
    return (
      <div className="history-widget explain-widget">
        <div className="history-widget-head">
          <h4>Explanation overlay</h4>
          <div className="explain-head-right">
            <span className="explain-head-note">
              Highlights areas influencing the prediction
            </span>
            {adminMode && (
              <button
                type="button"
                className="ghost-btn"
                onClick={() => navigate(`/admin/annotate/${scanId}`)}
                style={{ fontSize: 12, padding: "7px 10px" }}
              >
                Annotate
              </button>
            )}
          </div>
        </div>

        {!explainTarget && !explainLoading && (
          <p className="hint explain-status">Tap a percentage signal above to load its overlay.</p>
        )}

        {explainLoading && (
          <p className="hint explain-status">
            Generating overlay…
          </p>
        )}

        {explainError && (
          <p className="error-msg explain-status">
            {explainError}
          </p>
        )}

        {explainOverlay && (
          <div className="explain-overlay-wrap">
            {typeof explainOverlay === "string" ? (
              // Overlay type: base64 PNG (Grad-CAM)
              <>
                <div className="explain-media-frame">
                  <img
                    src={`data:image/png;base64,${explainOverlay}`}
                    alt="Explainability overlay"
                    className="explain-overlay-img"
                  />
                </div>
                <p className="explain-caption">
                  Warmer (redder) regions contributed more to the selected prediction.
                </p>
              </>
            ) : explainOverlay?.type === "boxes" ? (
              // Boxes type: YOLO detections with boxed image
              <div className="explain-boxes">
                <p className="explain-boxes-title">
                  <strong>{prettyName[explainTarget] || explainTarget}</strong>
                  {" "}
                  • {explainOverlay.detection_count || 0} localized detections
                </p>
                
                {/* Display boxed image if available */}
                {explainOverlay.image_png_base64 && (
                  <div className="explain-media-frame">
                    <img
                      src={`data:image/png;base64,${explainOverlay.image_png_base64}`}
                      alt={`${explainTarget} detections`}
                      className="explain-overlay-img explain-boxes-img"
                    />
                  </div>
                )}
                
                {explainOverlay.detections?.length === 0 ? (
                  <p className="explain-caption">
                    No localized findings detected. This condition may be diffuse or not present at this confidence level.
                  </p>
                ) : (
                  <ul className="explain-detection-list">
                    {explainOverlay.detections.map((d, i) => (
                      <li key={i}>
                        <strong>{d.label}</strong> — {(d.confidence * 100).toFixed(1)}% confidence
                      </li>
                    ))}
                  </ul>
                )}
                <p className="explain-method-note">
                  Detection-based localization (YOLO object detector).
                </p>
              </div>
            ) : null}
          </div>
        )}
      </div>
    );
  };

  const OutcomeSimulationCard = ({ scanId }) => {
    const data = simCache[scanId];
    const isOpen = simOpenForScan === scanId;
    const isBusy = simLoading && isOpen;
    const scanError = isOpen ? simError : "";

    return (
      <div className="history-widget outcome-widget">
        <div className="history-widget-head">
          <h4>Expected outcome preview</h4>
          <button
            type="button"
            className="ghost-btn"
            onClick={() => handleOutcomeToggle(scanId)}
            style={{ fontSize: 12, padding: "7px 10px" }}
          >
            {isOpen ? "Hide preview" : "Generate preview"}
          </button>
        </div>

        {!isOpen && (
          <p className="hint explain-status">
            Generate a visual projection of improvement based on this scan.
          </p>
        )}

        {isBusy && (
          <p className="hint explain-status">Generating expected outcome…</p>
        )}

        {scanError && <p className="error-msg explain-status">{scanError}</p>}

        {isOpen && data && (
          <div className="outcome-compare">
            <div className="outcome-col">
              <p className="outcome-label">Current</p>
              <div className="explain-media-frame">
                <img
                  src={`data:image/png;base64,${data.current_png_base64}`}
                  alt="Current face"
                  className="explain-overlay-img"
                />
              </div>
            </div>
            <div className="outcome-col">
              <p className="outcome-label">Expected</p>
              <div className="explain-media-frame">
                <img
                  src={`data:image/png;base64,${data.expected_png_base64}`}
                  alt="Expected outcome preview"
                  className="explain-overlay-img"
                />
              </div>
            </div>
            <p className="outcome-disclaimer">
              {data.disclaimer || "Visual simulation only. Not a diagnosis."}
              {typeof data?.meta?.mean_delta === "number"
                ? ` • effect strength: ${(data.meta.mean_delta * 100).toFixed(1)}%`
                : ""}
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-root">
      <AppHeader />

      <main className="app-main history-page-main">
        <section className="card dashboard-card history-page-card">
          <div className="history-page-head">
            <div>
              <h2>{title}</h2>
              <p className="history-page-subtitle">
                Your scan timeline with confidence signals and AI evidence.
              </p>
            </div>
            <div className="history-head-right">
              <span className="history-head-badge">Skin Log</span>
              {!loading && (
                <span className="history-count-pill">
                  {scans.length} {scans.length === 1 ? "scan" : "scans"}
                </span>
              )}
            </div>
          </div>
          {adminMode && targetUser && (
            <p className="hint history-admin-user">
              {targetUser.name || "User"} ({targetUser.email}) - role: {String(targetUser.role || "user").toUpperCase()}
            </p>
          )}
          {adminMode && userIdOverride && (
            <div className="history-admin-back">
              <button className="ghost-btn" onClick={() => navigate("/admin/users")}>
                Back to users
              </button>
            </div>
          )}

          {loading && <p className="hint">Loading your scan history…</p>}

          {!loading && scans.length === 0 && (
            <p className="hint">You don&apos;t have any saved scans yet.</p>
          )}

          {!loading && scans.length > 0 && (
            <div className="history-list">
              {scans.map((scan) => {
                const { concerns, skinType, routine, outcome, risk } = unpack(scan);
                const isOpen = openId === scan.id;
                const createdAt = new Date(scan.created_at);
                const weekday = createdAt.toLocaleDateString(undefined, { weekday: "long" });
                const shortDate = createdAt.toLocaleDateString(undefined, {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                });
                const previewUrl = resolveImageUrl(
                  scan.image_path_face_raw ||
                    scan.image_path_face ||
                    scan.image_path_full ||
                    scan.image_path
                );

                return (
                  <article key={scan.id} className={"history-entry" + (isOpen ? " is-open" : "")}>
                    {/* MAIN SCAN ROW */}
                    <div
                      className="history-item"
                      onClick={() => toggle(scan.id)}
                    >
                      <div className="history-item-shell">
                        <div className="history-col-meta">
                          <div className="history-chrono-row">
                            <span className="history-chrono-dot" />
                            <span className="history-chrono-weekday">{weekday}</span>
                            <span className="history-chrono-date">{shortDate}</span>
                          </div>

                          <div className="history-header">
                            <div className="history-meta">
                              <span className="history-date">
                                {new Date(scan.created_at).toLocaleString()}
                              </span>
                              <span className="history-id">Scan #{scan.id}</span>
                            </div>
                            <span className="history-toggle-tag">
                              <span>{isOpen ? "Hide details" : "View details"}</span>
                              <span className={"history-toggle-chevron" + (isOpen ? " is-open" : "")}>
                                ▾
                              </span>
                            </span>
                          </div>

                          {skinType && (
                            <div className="history-skin-row">
                              <span>Skin type:</span> <b>{skinType.skin_type?.toUpperCase()}</b>
                              <em>({((skinType.confidence || 0) * 100).toFixed(1)}%)</em>
                            </div>
                          )}

                          {previewUrl && (
                            <div className="history-preview-row">
                              <div className="history-mini-preview">
                                <img src={previewUrl} alt="Scan capture preview" loading="lazy" />
                              </div>
                              <div className="history-preview-copy">
                                <span className="history-preview-label">Capture preview</span>
                                <span className="history-preview-meta">Front view</span>
                              </div>
                            </div>
                          )}

                          <span className="history-meta-note">
                            Tap to {isOpen ? "collapse" : "expand"} details
                          </span>
                        </div>

                        <div className="history-col-signals">
                          <div className="history-signals">
                            {concerns &&
                              Object.entries(concerns).map(([key, value]) => {
                                const prob = value?.probability || 0;
                                const present = prob >= UI_DETECT_THRESHOLD;
                                const percent = (prob * 100).toFixed(1);

                                return (
                                  <button
                                    key={key}
                                    type="button"
                                    className={
                                      "history-signal history-signal-btn " +
                                      (present ? "history-signal-present" : "history-signal-absent")
                                    }
                                    onClick={(e) => handleSignalExplain(e, scan.id, key)}
                                  >
                                    <div className="history-signal-top">
                                      <span className="history-signal-name">
                                        {prettyName[key] || key}
                                      </span>
                                      <span className="history-signal-percent">{percent}%</span>
                                    </div>
                                    <div className="history-signal-bar">
                                      <div
                                        className="history-signal-fill"
                                        style={{ width: `${Math.max(0, Math.min(prob * 100, 100))}%` }}
                                      />
                                    </div>
                                  </button>
                                );
                              })}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* EXPANDED DETAILS PANEL */}
                    <div className={"history-detail-wrap" + (isOpen ? " is-open" : "")}>
                      <div className="history-detail-panel">
                        <div className="history-detail-head">
                          <span className="history-detail-kicker">Scan breakdown</span>
                          <span className="history-detail-time">
                            {new Date(scan.created_at).toLocaleString()}
                          </span>
                        </div>
                        <div className="history-detail-content">
                          {/* Phase 4A: Explainability */}
                          <ExplainabilityCard scanId={scan.id} />
                          <OutcomeSimulationCard scanId={scan.id} />

                          {/* Phase 3: Risk/Benefit */}
                          <RiskBenefitWidget outcome={outcome} risk={risk} />

                          {/* Routine / formulas (admin only) */}
                          {adminMode && routine && (
                            <>
                              <h4 className="detail-section-title">Generated routine & formulas</h4>
                              <p className="detail-subhint">
                                Scroll inside this box to view full formula ↓
                              </p>

                              {routine.steps &&
                                routine.steps.map((step) => (
                                  <div
                                    key={step.step}
                                    className="detail-card"
                                  >
                                    <div className="detail-card-title">
                                      {String(step.step || "").toUpperCase()}
                                    </div>

                                    {step.changes?.length > 0 && (
                                      <ul className="detail-change-list">
                                        {step.changes.map((c, i) => (
                                          <li key={i}>
                                            {c.inci}: {c.from}% → {c.to}%
                                          </li>
                                        ))}
                                      </ul>
                                    )}

                                    {step.formula_personalized && (
                                      <table className="detail-table">
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

                          {adminMode && scan.annotation && (
                            <div className="detail-card annotation-card">
                              <h4 className="detail-section-title no-top">Saved annotations</h4>
                              <div className="detail-subhint tight">
                                Boxes: {Array.isArray(scan.annotation.annotations?.boxes) ? scan.annotation.annotations.boxes.length : 0}
                                {scan.annotation.updated_at ? ` | Updated: ${new Date(scan.annotation.updated_at).toLocaleString()}` : ""}
                              </div>
                              {scan.annotation.notes && (
                                <p className="annotation-notes">
                                  <b>Notes:</b> {scan.annotation.notes}
                                </p>
                              )}
                            </div>
                          )}

                          {!adminMode && !outcome && !risk && (
                            <p className="hint" style={{ marginTop: 0 }}>
                              No additional data stored for this scan (older entry).
                            </p>
                          )}

                          {adminMode && !routine && !outcome && !risk && (
                            <p className="hint" style={{ marginTop: 0 }}>
                              No routine/risk data stored for this scan (older entry).
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  </article>
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
