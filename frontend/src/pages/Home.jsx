import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import "../App.css";
import AppHeader from "../components/AppHeader";
import API_BASE from "../config";

function useScrollReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".sr");
    if (!els.length) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("sr-visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0 }
    );
    els.forEach((el) => io.observe(el));
    const fallback = setTimeout(() => {
      els.forEach((el) => el.classList.add("sr-visible"));
    }, 1500);
    return () => {
      io.disconnect();
      clearTimeout(fallback);
    };
  }, []);
}

function Home() {
  useScrollReveal();

  const stored = localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
  let user = null;
  try { user = stored ? JSON.parse(stored) : null; } catch { user = null; }

  const [hasPreviousScans, setHasPreviousScans] = useState(false);
  const userId = useMemo(() => user?.id ?? null, [user]);

  useEffect(() => {
    let mounted = true;
    const checkScans = async () => {
      if (!userId) { if (mounted) setHasPreviousScans(false); return; }
      try {
        const res = await axios.get(`${API_BASE}/scans`, { params: { user_id: userId, limit: 1 } });
        if (mounted) setHasPreviousScans(Array.isArray(res.data?.items) && res.data.items.length > 0);
      } catch { if (mounted) setHasPreviousScans(false); }
    };
    checkScans();
    return () => { mounted = false; };
  }, [userId]);

  const isAdmin = String(user?.role || "").toLowerCase() === "admin";
  const ctaLabel = user ? "Start a new scan" : "Get started — it's free";
  const ctaLink = user ? "/scan" : "/login";

  return (
    <div className="app-root">
      <AppHeader />

      {/* Hero lives OUTSIDE home-main so it gets full width natively */}
      <section className="home-hero-card">
        <div className="home-hero-inner">
          <div className="home-hero-left">
            <p className="home-hero-kicker">Skin analysis, personalised for you</p>
            <h2>
              Know exactly what
              <br />
              your skin needs
            </h2>
            <p className="home-hero-text">
              Take one selfie and get a personalised breakdown of your skin in seconds —
              acne, redness, hyperpigmentation and more, with a routine built around your results.
            </p>

            <div className="home-hero-actions">
              <Link to={ctaLink} className="primary-btn home-main-cta">{ctaLabel}</Link>
              {isAdmin && (
                <Link to="/admin/users" className="ghost-btn home-secondary-cta">Admin panel</Link>
              )}
              {user && hasPreviousScans && (
                <Link to="/history" className="ghost-btn home-secondary-cta">View recent scans</Link>
              )}
            </div>

            <div className="home-hero-stats">
              <div className="home-hero-stat"><strong>5</strong><span>Skin conditions</span></div>
              <div className="home-hero-stat"><strong>2.5s</strong><span>Scan time</span></div>
              <div className="home-hero-stat"><strong>100%</strong><span>Free to use</span></div>
            </div>

            <div className="home-hero-badges">
              <span>No sign-up fee</span>
              <span>Personalised routine</span>
              <span>Progress tracking</span>
            </div>
          </div>

          <div className="home-hero-right">
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
                    <div className="home-mini-row"><span>Acne</span><span>32%</span></div>
                    <div className="home-mini-row"><span>Redness</span><span>11%</span></div>
                    <div className="home-mini-row"><span>Blackheads</span><span>24%</span></div>
                  </div>
                  <div className="home-screen-footer">
                    <span>Recent trend</span>
                    <span className="home-trend-pill">Improving</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <main className="home-main">
        <section className="home-neutral-band">
          <section className="home-section-wrap">
            <div className="home-section-head sr sr-up">
              <h3>Why AURAI?</h3>
              <p>Real analysis, honest results, and a routine that actually fits your skin.</p>
            </div>
            <div className="home-feature-grid">
              {[
                { icon: "◎", title: "Instant skin analysis", text: "Upload a photo and get a clear breakdown of acne, redness, blackheads and more in seconds." },
                { icon: "✦", title: "Your personalised routine", text: "Step-by-step skincare suggestions tailored to what your skin actually needs right now." },
                { icon: "▢", title: "Clear next steps", text: "Know exactly what to use, when to use it, and why — no guesswork." },
                { icon: "↗", title: "Track your progress", text: "Save your scans and watch how your skin changes over time." },
              ].map((f, i) => (
                <article key={i} className="home-feature-card sr sr-up" style={{ "--sr-delay": `${i * 0.1}s` }}>
                  <div className="home-feature-icon">{f.icon}</div>
                  <h4>{f.title}</h4>
                  <p>{f.text}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="home-section-wrap">
            <div className="home-section-head sr sr-up">
              <h3>How it works</h3>
              <p>Three steps from photo to personalised routine — takes under a minute.</p>
            </div>
            <div className="home-stair-grid">
              {[
                { n: "1", title: "Take a photo", text: "Upload a selfie or use your camera. We check the lighting and clarity automatically.", side: "left" },
                { n: "2", title: "We analyse your skin", text: "Our AI scans for five conditions and highlights exactly where concerns appear on your face.", side: "right" },
                { n: "3", title: "Get your routine", text: "Receive a personalised skincare routine based on your results, saved for next time.", side: "left" },
              ].map((s, i) => (
                <article key={i} className={`home-stair-card sr sr-${s.side}`} style={{ "--sr-delay": `${i * 0.15}s` }}>
                  <span className="home-flow-number">{s.n}</span>
                  <div>
                    <h4>{s.title}</h4>
                    <p>{s.text}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </section>

        <section className="home-cta-strip sr sr-up">
          <h3>Your skin deserves better</h3>
          <p>Take a free scan and find out exactly what your skin needs today.</p>
          <Link to={ctaLink} className="home-cta-strip-btn">Start your free scan</Link>
        </section>
      </main>

      <footer className="home-footer">
        <div className="home-footer-top">
          <div className="home-footer-brand">
            <h4>AURAI</h4>
            <p>Personalised skin analysis in seconds — built to help you actually understand and take care of your skin.</p>
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
