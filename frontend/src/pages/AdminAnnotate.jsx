import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import axios from "axios";
import AppHeader from "../components/AppHeader";
import "../App.css";

const API_BASE = "http://127.0.0.1:8000";

const LABELS = [
  "ignore_moustache",
  "ignore_lips",
  "ignore_eyebrows",
  "ignore_hairline",
  "ignore_mole",
  "acne_spot",
  "blackhead_cluster",
  "bags_region",
  "redness_region",
  "hyperpigmentation_region",
];

function normalizeBox(box, imgW, imgH) {
  let { x1, y1, x2, y2 } = box;
  x1 = Math.max(0, Math.min(imgW - 1, Math.round(x1)));
  y1 = Math.max(0, Math.min(imgH - 1, Math.round(y1)));
  x2 = Math.max(0, Math.min(imgW, Math.round(x2)));
  y2 = Math.max(0, Math.min(imgH, Math.round(y2)));
  if (x2 <= x1) x2 = Math.min(imgW, x1 + 1);
  if (y2 <= y1) y2 = Math.min(imgH, y1 + 1);
  return { x1, y1, x2, y2 };
}

export default function AdminAnnotate() {
  const navigate = useNavigate();
  const { scanId } = useParams();
  const containerRef = useRef(null);
  const [currentUser, setCurrentUser] = useState(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const [imageB64, setImageB64] = useState("");
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 });
  const [displaySize, setDisplaySize] = useState({ w: 1, h: 1 });

  const [label, setLabel] = useState(LABELS[0]);
  const [tool, setTool] = useState("box"); // box | free
  const [notes, setNotes] = useState("");
  const [boxes, setBoxes] = useState([]);
  const [strokes, setStrokes] = useState([]);

  const [draft, setDraft] = useState(null);
  const [draftStroke, setDraftStroke] = useState(null);
  const [savedMsg, setSavedMsg] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem("auraiUser") || sessionStorage.getItem("auraiUser");
    if (!stored) {
      navigate("/login", { replace: true });
      return;
    }
    let user = null;
    try {
      user = JSON.parse(stored);
    } catch {
      navigate("/login", { replace: true });
      return;
    }
    if (String(user?.role || "user").toLowerCase() !== "admin") {
      navigate("/history", { replace: true });
      return;
    }
    setCurrentUser(user);

    const load = async () => {
      try {
        setLoading(true);
        setError("");

        const [imgRes, annRes] = await Promise.all([
          axios.get(`${API_BASE}/admin/scan-image/${scanId}`, {
            params: { kind: "face_raw", user_id: user.id },
          }),
          axios.get(`${API_BASE}/admin/annotations`, {
            params: { scan_id: Number(scanId), user_id: user.id },
          }),
        ]);

        const img = imgRes.data;
        setImageB64(img.image_png_base64 || "");
        setImgSize(img.size || { w: 1, h: 1 });

        const ann = annRes.data?.annotation;
        const annBoxes = ann?.annotations?.boxes || [];
        const annStrokes = ann?.annotations?.strokes || [];
        setBoxes(Array.isArray(annBoxes) ? annBoxes : []);
        setStrokes(Array.isArray(annStrokes) ? annStrokes : []);
        setNotes(ann?.notes || "");
      } catch (e) {
        console.error(e);
        setError(e?.response?.data?.detail || "Failed to load annotation data.");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [navigate, scanId]);

  const scale = useMemo(() => {
    return {
      sx: displaySize.w / Math.max(1, imgSize.w),
      sy: displaySize.h / Math.max(1, imgSize.h),
    };
  }, [displaySize, imgSize]);

  const toImageCoords = (clientX, clientY) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    const x = (clientX - rect.left) / Math.max(1e-6, scale.sx);
    const y = (clientY - rect.top) / Math.max(1e-6, scale.sy);
    return {
      x: Math.max(0, Math.min(imgSize.w, x)),
      y: Math.max(0, Math.min(imgSize.h, y)),
    };
  };

  const onPointerDown = (e) => {
    if (!imageB64) return;
    e.preventDefault();
    setSavedMsg("");
    if (containerRef.current && e.pointerId != null) {
      try {
        containerRef.current.setPointerCapture(e.pointerId);
      } catch {}
    }
    const p = toImageCoords(e.clientX, e.clientY);
    if (tool === "free") {
      setDraftStroke({
        label,
        points: [[p.x, p.y]],
        width: 3,
        source: "admin",
      });
      setDraft(null);
    } else {
      setDraft({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
      setDraftStroke(null);
    }
  };

  const onPointerMove = (e) => {
    e.preventDefault();
    const p = toImageCoords(e.clientX, e.clientY);
    if (tool === "free") {
      if (!draftStroke) return;
      setDraftStroke((s) => ({
        ...s,
        points: [...s.points, [p.x, p.y]],
      }));
      return;
    }
    if (!draft) return;
    setDraft((d) => ({ ...d, x2: p.x, y2: p.y }));
  };

  const onPointerUp = (e) => {
    if (e) e.preventDefault();
    if (tool === "free") {
      if (!draftStroke) return;
      if ((draftStroke.points || []).length >= 2) {
        setStrokes((prev) => [...prev, draftStroke]);
      }
      setDraftStroke(null);
      return;
    }
    if (!draft) return;
    const box = normalizeBox(draft, imgSize.w, imgSize.h);
    const area = (box.x2 - box.x1) * (box.y2 - box.y1);
    if (area >= 16) {
      setBoxes((prev) => [
        ...prev,
        { label, box: [box.x1, box.y1, box.x2, box.y2], conf: 1.0, source: "admin" },
      ]);
    }
    setDraft(null);
  };

  const removeBox = (idx) => {
    setBoxes((prev) => prev.filter((_, i) => i !== idx));
  };
  const removeStroke = (idx) => {
    setStrokes((prev) => prev.filter((_, i) => i !== idx));
  };

  const save = async () => {
    try {
      setSaving(true);
      setError("");
      const user = currentUser;

      await axios.post(`${API_BASE}/admin/annotations`, {
        scan_id: Number(scanId),
        user_id: user?.id || null,
        image_kind: "face_raw",
        boxes,
        strokes,
        notes,
      });
      setSavedMsg("Annotations saved.");
    } catch (e) {
      console.error(e);
      setError(e?.response?.data?.detail || "Failed to save annotations.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="app-root">
      <AppHeader />
      <main className="app-main">
        <section className="card dashboard-card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>Admin Annotation - Scan #{scanId}</h2>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="ghost-btn" onClick={() => navigate("/admin/history")}>Back</button>
              <button className="primary-btn" onClick={save} disabled={saving || loading}>
                {saving ? "Saving..." : "Save annotations"}
              </button>
            </div>
          </div>
          {savedMsg && <p className="hint" style={{ marginTop: 0 }}>{savedMsg}</p>}

          <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 14 }}>
            <div style={{ background: "#fff", border: "1px solid #ece8f8", borderRadius: 12, padding: 12 }}>
              <label style={{ fontWeight: 700, fontSize: 13 }}>Label</label>
              <select
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                style={{ width: "100%", marginTop: 6, padding: 8, borderRadius: 8 }}
              >
                {LABELS.map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
              <label style={{ display: "block", marginTop: 12, fontWeight: 700, fontSize: 13 }}>Marker tool</label>
              <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                <button
                  type="button"
                  className={tool === "box" ? "primary-btn" : "ghost-btn"}
                  onClick={() => setTool("box")}
                >
                  Box
                </button>
                <button
                  type="button"
                  className={tool === "free" ? "primary-btn" : "ghost-btn"}
                  onClick={() => setTool("free")}
                >
                  Free marker
                </button>
              </div>

              <label style={{ display: "block", marginTop: 12, fontWeight: 700, fontSize: 13 }}>Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={4}
                style={{ width: "100%", marginTop: 6, padding: 8, borderRadius: 8 }}
              />

              <p className="hint" style={{ marginTop: 10 }}>
                Use Box for precise regions or Free marker for brush-style regions. Both are saved for training feedback.
              </p>

              <div style={{ marginTop: 10, maxHeight: 340, overflowY: "auto", borderTop: "1px solid #eee", paddingTop: 8 }}>
                {boxes.length === 0 && strokes.length === 0 && (
                  <p className="hint" style={{ margin: 0 }}>No markers yet.</p>
                )}
                {boxes.map((b, i) => (
                  <div key={`box-${b.label}-${i}`} style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
                    <div style={{ fontSize: 12 }}>
                      <b>[BOX] {b.label}</b><br />
                      <span style={{ opacity: 0.7 }}>{b.box.join(", ")}</span>
                    </div>
                    <button className="ghost-btn" onClick={() => removeBox(i)}>Remove</button>
                  </div>
                ))}
                {strokes.map((s, i) => (
                  <div key={`stroke-${s.label}-${i}`} style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
                    <div style={{ fontSize: 12 }}>
                      <b>[FREE] {s.label}</b><br />
                      <span style={{ opacity: 0.7 }}>{(s.points || []).length} points</span>
                    </div>
                    <button className="ghost-btn" onClick={() => removeStroke(i)}>Remove</button>
                  </div>
                ))}
              </div>
            </div>

            <div>
              {loading && <p className="hint">Loading image...</p>}
              {error && <p className="error-msg">{error}</p>}

              {!!imageB64 && (
                <div
                  ref={containerRef}
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                  onPointerCancel={onPointerUp}
                  style={{
                    position: "relative",
                    display: "inline-block",
                    border: "1px solid rgba(0,0,0,0.12)",
                    borderRadius: 10,
                    cursor: "crosshair",
                    userSelect: "none",
                    maxWidth: "100%",
                    overflow: "hidden",
                    touchAction: "none",
                  }}
                >
                  <img
                    src={`data:image/png;base64,${imageB64}`}
                    alt="scan"
                    style={{ display: "block", maxWidth: "100%", height: "auto" }}
                    draggable={false}
                    onLoad={(e) => {
                      const el = e.currentTarget;
                      setDisplaySize({ w: el.clientWidth, h: el.clientHeight });
                    }}
                  />

                  {boxes.map((b, idx) => {
                    const [x1, y1, x2, y2] = b.box;
                    const left = x1 * scale.sx;
                    const top = y1 * scale.sy;
                    const w = (x2 - x1) * scale.sx;
                    const h = (y2 - y1) * scale.sy;
                    return (
                      <div
                        key={`box-${idx}`}
                        style={{
                          position: "absolute",
                          left,
                          top,
                          width: Math.max(1, w),
                          height: Math.max(1, h),
                          border: "2px solid #ff4d4f",
                          background: "rgba(255, 77, 79, 0.12)",
                          pointerEvents: "none",
                        }}
                        title={b.label}
                      >
                        <span style={{ fontSize: 10, background: "#ff4d4f", color: "#fff", padding: "1px 4px", pointerEvents: "none" }}>{b.label}</span>
                      </div>
                    );
                  })}
                  <svg
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      width: displaySize.w,
                      height: displaySize.h,
                      pointerEvents: "none",
                    }}
                    viewBox={`0 0 ${Math.max(1, displaySize.w)} ${Math.max(1, displaySize.h)}`}
                  >
                    {strokes.map((s, idx) => {
                      const pts = (s.points || [])
                        .map(([x, y]) => `${x * scale.sx},${y * scale.sy}`)
                        .join(" ");
                      if (!pts) return null;
                      return (
                        <polyline
                          key={`stroke-poly-${idx}`}
                          points={pts}
                          fill="none"
                          stroke="#00b894"
                          strokeWidth={Math.max(2, (s.width || 3))}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          opacity="0.95"
                        />
                      );
                    })}
                    {draftStroke && (
                      <polyline
                        points={(draftStroke.points || [])
                          .map(([x, y]) => `${x * scale.sx},${y * scale.sy}`)
                          .join(" ")}
                        fill="none"
                        stroke="#7b61ff"
                        strokeWidth={3}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity="0.9"
                      />
                    )}
                  </svg>

                  {draft && (() => {
                    const x1 = Math.min(draft.x1, draft.x2);
                    const y1 = Math.min(draft.y1, draft.y2);
                    const x2 = Math.max(draft.x1, draft.x2);
                    const y2 = Math.max(draft.y1, draft.y2);
                    return (
                      <div
                        style={{
                          position: "absolute",
                          left: x1 * scale.sx,
                          top: y1 * scale.sy,
                          width: (x2 - x1) * scale.sx,
                          height: (y2 - y1) * scale.sy,
                          border: "2px dashed #7b61ff",
                          background: "rgba(123, 97, 255, 0.12)",
                          pointerEvents: "none",
                        }}
                      />
                    );
                  })()}
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
