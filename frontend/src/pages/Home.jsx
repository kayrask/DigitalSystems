import React from "react";
import { Link } from "react-router-dom";
import "../App.css";
import AppHeader from "../components/AppHeader";

function Home() {
  const stored =
    localStorage.getItem("auraiUser") ||
    sessionStorage.getItem("auraiUser");

  let user = null;
  try {
    user = stored ? JSON.parse(stored) : null;
  } catch {
    user = null;
  }

  const ctaLabel = user ? "✨ Start a new scan" : "✨ Get started";
  const ctaLink = user ? "/scan" : "/login";

  return (
    <div className="app-root home-theme">
      <AppHeader variant="hero" />

      <main className="home-main">
        <section className="home-hero-shell">
          <div className="home-hero-left editorial">
            <h2>AI Skin Analysis Companion</h2>
            <p className="home-hero-text">
              A professional skin insights app built from your own AI pipeline.
              Scan, track trends, review localized findings, and personalize care
              with safer recommendations.
            </p>

            <div className="home-hero-actions">
              <Link to={ctaLink} className="primary-btn home-main-cta">
                {ctaLabel}
              </Link>
            </div>

            <div className="home-stats-row">
              <div className="home-stat">
                <b>30+</b>
                <span>Regression cases</span>
              </div>
              <div className="home-stat">
                <b>5</b>
                <span>Core conditions</span>
              </div>
              <div className="home-stat">
                <b>ROI+YOLO</b>
                <span>Hybrid explainability</span>
              </div>
            </div>
          </div>

          <div className="home-hero-right phone-wrap">
            <div className="home-phone premium">
              <div className="home-phone-header">AURAI Live View</div>
              <div className="home-phone-screen">
                <div className="home-face-placeholder">Face scan preview</div>
                <div className="home-mini-results">
                  <div className="home-mini-row">
                    <span>Acne</span>
                    <span>29.8%</span>
                  </div>
                  <div className="home-mini-row">
                    <span>Hyperpigmentation</span>
                    <span>2.0%</span>
                  </div>
                  <div className="home-mini-row">
                    <span>Blackheads</span>
                    <span>1.8%</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="leaf leaf-a" />
            <div className="leaf leaf-b" />
          </div>
        </section>

        <section className="home-grid">
          <div className="card">
            <h2>How it works</h2>
            <p className="home-section-text">
              1. Create an account and log in. <br />
              2. Take or upload a clear, front-facing photo. <br />
              3. AURAI estimates the visibility of several skin conditions and
              shows them as percentages.
            </p>
          </div>

          <div className="card">
            <h2>What it can see</h2>
            <p className="home-section-text">
              The current model focuses on acne, redness, blackheads,
              under-eye bags and hyperpigmentation. Results can change
              with lighting, camera quality, makeup and angle.
            </p>
          </div>

          <div className="card">
            <h2>Why this exists</h2>
            <p className="home-section-text">
              This prototype is part of a university dissertation exploring
              AI-assisted skin analysis. It is meant to support reflection and
              tracking, not to replace professionals or real diagnosis.
            </p>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>© 2025 AURAI – Research prototype only.</p>
      </footer>
    </div>
  );
}

export default Home;
