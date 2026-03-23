import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";
import axios from "axios";
import API_BASE from "../config";

import cleanserImg from "../assets/routine/cleanser.png";
import tonerImg from "../assets/routine/toner.png";
import serumImg from "../assets/routine/serum.png";
import moisturiserImg from "../assets/routine/moisturiser.png";

const STEP_META = {
  cleanser: {
    label: "Cleanser",
    step: "Step 1",
    tagline: "Deep clean, balanced pH",
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    accentColor: "#667eea",
    image: cleanserImg,
  },
  toner: {
    label: "Toner",
    step: "Step 2",
    tagline: "Rebalance & prep skin",
    gradient: "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    accentColor: "#a18cd1",
    image: tonerImg,
  },
  serum: {
    label: "Serum",
    step: "Step 3",
    tagline: "Targeted active treatment",
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    accentColor: "#f093fb",
    image: serumImg,
  },
  moisturiser: {
    label: "Moisturiser",
    step: "Step 4",
    tagline: "Seal, protect & hydrate",
    gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    accentColor: "#4facfe",
    image: moisturiserImg,
  },
};

const CONCERN_LABELS = {
  acne: "Acne",
  bags: "Under-eye bags",
  blackheads: "Blackheads",
  hyperpigmentation: "Hyperpigmentation",
  redness: "Redness",
};

function getTopIngredients(formula, n = 6) {
  return [...formula]
    .filter((r) => r.inci && r.percent != null)
    .sort((a, b) => b.percent - a.percent)
    .slice(0, n);
}

export default function Routine() {
  const { scanId } = useParams();
  const navigate = useNavigate();
  const [scan, setScan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState({});

  useEffect(() => {
    if (!scanId) return;
    axios
      .get(`${API_BASE}/scans/${scanId}`)
      .then((res) => {
        setScan(res.data);
        setLoading(false);
      })
      .catch(() => {
        setError("Could not load this routine.");
        setLoading(false);
      });
  }, [scanId]);

  const routine = scan?.results?.routine;
  const skinType = scan?.results?.skin_type?.skin_type || "combination";
  const concerns = scan?.results?.results || {};
  const detectedConcerns = Object.entries(concerns)
    .filter(([, v]) => (v?.probability || 0) >= 0.5)
    .map(([k]) => k);

  const handleSave = (step) => {
    setSaved((prev) => ({ ...prev, [step]: true }));
  };

  return (
    <div className="app-root">
      <AppHeader />
      <main className="app-main routine-main">
        {/* Page header */}
        <div className="routine-page-header">
          <button className="routine-back-btn" onClick={() => navigate(-1)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Back
          </button>
          <div className="routine-page-title-row">
            <div>
              <h1 className="routine-page-title">Your Personalised Routine</h1>
              {scan?.created_at && (
                <p className="routine-page-subtitle">
                  Generated from scan on{" "}
                  {new Date(scan.created_at).toLocaleDateString("en-GB", {
                    day: "numeric", month: "long", year: "numeric",
                  })}
                </p>
              )}
            </div>
            <span className="routine-skin-badge">
              {skinType.charAt(0).toUpperCase() + skinType.slice(1)} skin
            </span>
          </div>

          {detectedConcerns.length > 0 && (
            <div className="routine-concern-banner">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" />
              </svg>
              <span>
                Formula personalised for{" "}
                <strong>
                  {detectedConcerns.map((c) => CONCERN_LABELS[c] || c).join(", ")}
                </strong>
              </span>
            </div>
          )}
        </div>

        {loading && (
          <div className="routine-loading">
            <div className="routine-spinner" />
            <p>Loading your routine…</p>
          </div>
        )}

        {error && <p className="routine-error">{error}</p>}

        {!loading && !error && routine?.steps && (
          <div className="routine-steps">
            {routine.steps.map((stepData) => {
              const meta = STEP_META[stepData.step] || STEP_META.cleanser;
              const topIngredients = getTopIngredients(
                stepData.formula_personalized || stepData.formula_base || []
              );
              const changes = stepData.changes || [];
              const isSaved = saved[stepData.step];

              return (
                <div key={stepData.step} className="routine-product-card">
                  {/* Product visual */}
                  <div
                    className="routine-product-image"
                    style={{ background: meta.gradient, "--routine-accent": meta.gradient }}
                  >
                    {meta.image && (
                      <img
                        src={meta.image}
                        alt={meta.label}
                        className="routine-product-img"
                      />
                    )}
                    <div className="routine-product-step-label">{meta.step}</div>
                  </div>

                  {/* Product info */}
                  <div className="routine-product-info">
                    <div className="routine-product-header">
                      <div>
                        <p className="routine-step-num">{meta.step}</p>
                        <h2 className="routine-product-name">
                          AURAI {meta.label}
                        </h2>
                        <p className="routine-product-tagline">{meta.tagline}</p>
                      </div>
                    </div>

                    {/* Targets */}
                    {stepData.focus_concerns?.length > 0 && (
                      <div className="routine-targets">
                        <span className="routine-section-label">Targets</span>
                        <div className="routine-target-chips">
                          {stepData.focus_concerns.slice(0, 3).map((c) => (
                            <span key={c} className="routine-target-chip">
                              {CONCERN_LABELS[c] || c}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Key ingredients */}
                    {topIngredients.length > 0 && (
                      <div className="routine-ingredients">
                        <span className="routine-section-label">Key ingredients</span>
                        <ul className="routine-ingredient-list">
                          {topIngredients.map((r) => (
                            <li key={r.inci} className="routine-ingredient-item">
                              <span className="routine-ingredient-dot" style={{ background: meta.accentColor }} />
                              <span className="routine-ingredient-name">{r.inci}</span>
                              {r.percent != null && (
                                <span className="routine-ingredient-pct">
                                  {r.percent.toFixed(1)}%
                                </span>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Personalisation tweaks */}
                    {changes.length > 0 && (
                      <div className="routine-tweaks">
                        <span className="routine-section-label routine-tweaks-label">
                          ✦ Personalised for you
                        </span>
                        <div className="routine-tweak-list">
                          {changes.slice(0, 3).map((c) => (
                            <div key={c.inci} className="routine-tweak-item">
                              <span className="routine-tweak-inci">{c.inci}</span>
                              <span className="routine-tweak-arrow">
                                {c.from.toFixed(2)}% → {c.to.toFixed(2)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* CTA */}
                    <button
                      className={"routine-save-btn" + (isSaved ? " saved" : "")}
                      onClick={() => handleSave(stepData.step)}
                    >
                      {isSaved ? (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                          Saved to routine
                        </>
                      ) : (
                        <>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                          </svg>
                          Save to my routine
                        </>
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {!loading && !error && !routine?.steps && (
          <div className="routine-empty">
            <p>No routine found for this scan.</p>
            <button className="routine-back-btn" onClick={() => navigate("/history")}>
              Back to history
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
