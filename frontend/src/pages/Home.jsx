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
    <div className="app-root">
      <AppHeader />

      <main className="home-main">
        {/* (rest of your Home JSX stays exactly as you pasted before) */}
        <section className="card home-hero-card">
          <div className="home-hero-left">
            <h2>Understand your skin, one scan at a time.</h2>
            <p className="home-hero-text">
              AURAI estimates acne, redness, blackheads and more from your facial
              photo using a deep learning model. It&apos;s designed to help you
              track trends in your skin rather than give a diagnosis.
            </p>

            <div className="home-hero-actions">
              <Link to={ctaLink} className="primary-btn home-main-cta">
                {ctaLabel}
              </Link>
            </div>

            
          </div>

          <div className="home-hero-right">
            <div className="home-phone">
              <div className="home-phone-header">Example scan</div>
              <div className="home-phone-screen">
                <div className="home-face-placeholder">Your face here</div>
                <div className="home-mini-results">
                  <div className="home-mini-row">
                    <span>Acne</span>
                    <span>82%</span>
                  </div>
                  <div className="home-mini-row">
                    <span>Redness</span>
                    <span>19%</span>
                  </div>
                  <div className="home-mini-row">
                    <span>Blackheads</span>
                    <span>37%</span>
                  </div>
                </div>
              </div>
              <div className="home-phone-footer">
                <span className="home-dot active" />
                <span className="home-dot" />
                <span className="home-dot" />
              </div>
            </div>
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
