import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import "../App.css";
import AppHeader from "../components/AppHeader";

import API_BASE from "../config";

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

  const [hasPreviousScans, setHasPreviousScans] = useState(false);

  const userId = useMemo(() => user?.id ?? null, [user]);

  useEffect(() => {
    let mounted = true;

    const checkScans = async () => {
      if (!userId) {
        if (mounted) setHasPreviousScans(false);
        return;
      }
      try {
        const res = await axios.get(`${API_BASE}/scans`, {
          params: { user_id: userId, limit: 1 },
        });
        const hasItems = Array.isArray(res.data?.items) && res.data.items.length > 0;
        if (mounted) setHasPreviousScans(hasItems);
      } catch {
        if (mounted) setHasPreviousScans(false);
      }
    };

    checkScans();
    return () => {
      mounted = false;
    };
  }, [userId]);

  const ctaLabel = user ? "✨ Start a new scan" : "✨ Get started";
  const ctaLink = user ? "/scan" : "/login";

  return (
    <div className="app-root">
      <AppHeader />

      <main className="home-main">
        <section className="home-hero-card">
          <div className="home-hero-left reveal reveal-1">
            <p className="home-hero-kicker reveal reveal-1">AI-Powered Skin Analysis</p>
            <h2 className="reveal reveal-1">
              Discover Your Perfect
              <br />
              Skincare Routine
            </h2>
            <p className="home-hero-text reveal reveal-2">
              Get personalized insights based on one clear selfie. AURAI combines
              model scoring, region-aware localization, and routine guidance in one flow.
            </p>

            <div className="home-hero-actions reveal reveal-3">
              <Link to={ctaLink} className="primary-btn home-main-cta">
                {ctaLabel}
              </Link>
              <Link to={user ? "/scan" : "/register"} className="ghost-btn home-secondary-cta">
                Browse features
              </Link>
              {user && hasPreviousScans && (
                <Link to="/history" className="ghost-btn home-secondary-cta">
                  View recent scans
                </Link>
              )}
            </div>

            <div className="home-hero-stats reveal reveal-4">
              <div className="home-hero-stat">
                <strong>5</strong><span>Core concerns tracked</span>
              </div>
              <div className="home-hero-stat">
                <strong>30+</strong><span>Regression checks</span>
              </div>
              <div className="home-hero-stat">
                <strong>2.5s</strong><span>Average scan time</span>
              </div>
            </div>

            <div className="home-hero-badges reveal reveal-4">
              <span>Free analysis</span>
              <span>Personalized output</span>
              <span>Explainable results</span>
            </div>
          </div>

          <div className="home-hero-right reveal reveal-4">
            <div className="home-visual-stage">
              <div className="home-portrait-backdrop" />
              <div className="home-phone">
                <div className="home-phone-notch" />
                <div className="home-phone-screen">
                  <div className="home-screen-hero">
                    <p>Scan confidence</p>
                    <strong>96.4%</strong>
                  </div>
                  <div className="home-mini-results">
                    <div className="home-mini-row">
                      <span>Acne</span>
                      <span>32%</span>
                    </div>
                    <div className="home-mini-row">
                      <span>Redness</span>
                      <span>11%</span>
                    </div>
                    <div className="home-mini-row">
                      <span>Blackheads</span>
                      <span>24%</span>
                    </div>
                  </div>
                  <div className="home-screen-footer">
                    <span>Recent trend</span>
                    <span className="home-trend-pill">Improving</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="home-neutral-band reveal reveal-2">
          <section className="home-section-wrap">
            <div className="home-section-head">
              <h3>Why choose AURAI?</h3>
              <p>Accurate analysis with personalized recommendations and progress tracking.</p>
            </div>
            <div className="home-feature-grid">
              <article className="home-feature-card">
                <div className="home-feature-icon">◎</div>
                <h4>AI Skin Analysis</h4>
                <p>Computer-vision models estimate concern visibility in seconds.</p>
              </article>
              <article className="home-feature-card">
                <div className="home-feature-icon">✦</div>
                <h4>Personalized Routine</h4>
                <p>Suggestions adapt to skin profile, confidence, and risk signals.</p>
              </article>
              <article className="home-feature-card">
                <div className="home-feature-icon">▢</div>
                <h4>Product Guidance</h4>
                <p>Structured outputs connect model evidence to practical next steps.</p>
              </article>
              <article className="home-feature-card">
                <div className="home-feature-icon">↗</div>
                <h4>Track Progress</h4>
                <p>Review previous scans and monitor trends over time.</p>
              </article>
            </div>
          </section>

          <section className="home-section-wrap reveal reveal-3">
            <div className="home-section-head">
              <h3>How it works</h3>
              <p>Three quick steps from upload to personalized guidance.</p>
            </div>
            <div className="home-flow-grid">
              <article className="home-flow-card">
                <span className="home-flow-number">1</span>
                <h4>Take a photo</h4>
                <p>Use upload or live camera with automatic quality checks.</p>
              </article>
              <article className="home-flow-card">
                <span className="home-flow-number">2</span>
                <h4>AI analysis</h4>
                <p>Classifier + ROI logic + detector fusion produce structured outputs.</p>
              </article>
              <article className="home-flow-card">
                <span className="home-flow-number">3</span>
                <h4>Get recommendations</h4>
                <p>Receive personalized routine suggestions and track progress.</p>
              </article>
            </div>
          </section>
        </section>

        <section className="home-cta-strip reveal reveal-4">
          <h3>Ready to transform your skin workflow?</h3>
          <p>Join users getting structured skin analysis and actionable guidance.</p>
          <Link to={ctaLink} className="home-cta-strip-btn">
            Start Your Free Scan
          </Link>
        </section>
      </main>

      <footer className="home-footer">
        <div className="home-footer-top">
          <div className="home-footer-brand">
            <h4>AURAI</h4>
            <p>
              AI-powered skin analysis and personalized routine support built for
              clear, trackable results.
            </p>
          </div>

          <div className="home-footer-links">
            <div>
              <h5>Product</h5>
              <Link to="/scan">Skin scan</Link>
              <Link to="/history">Recent scans</Link>
              <Link to={user ? "/scan" : "/register"}>How it works</Link>
            </div>
            <div>
              <h5>Company</h5>
              <span className="footer-link-placeholder">About</span>
              <span className="footer-link-placeholder">Research</span>
              <a href="mailto:support@aurai.app">Contact</a>
            </div>
            <div>
              <h5>Support</h5>
              <span className="footer-link-placeholder">Privacy</span>
              <span className="footer-link-placeholder">Terms</span>
              <a href="mailto:support@aurai.app">support@aurai.app</a>
            </div>
          </div>
        </div>

        <div className="home-footer-bottom">
          <p>© 2026 AURAI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default Home;
