import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import Webcam from "react-webcam";
import "../App.css";
import AppHeader from "../components/AppHeader";

const API_BASE = "http://127.0.0.1:8000";
const API_URL = `${API_BASE}/predict`;
const UI_DETECT_THRESHOLD = 0.5;

function Scan() {
  const navigate = useNavigate();

  const [mode, setMode] = useState("upload"); // "upload" | "camera"
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [results, setResults] = useState(null);     // skin concerns
  const [skinType, setSkinType] = useState(null);   // {skin_type, confidence, ...}

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const webcamRef = useRef(null);
  const [cameraGuide, setCameraGuide] = useState({
    ready: false,
    lighting: "unknown",
    lightingLevel: "warn",
    sharpness: "unknown",
    sharpnessLevel: "warn",
    contrast: "unknown",
    contrastLevel: "warn",
    tip: "Checking camera conditions...",
  });

  useEffect(() => {
    const storedUser =
      localStorage.getItem("auraiUser") ||
      sessionStorage.getItem("auraiUser");

    if (!storedUser) navigate("/login", { replace: true });
  }, [navigate]);

  const resetOutputs = () => {
    setResults(null);
    setSkinType(null);
    setError("");
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    resetOutputs();

    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const captureFromWebcam = () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    resetOutputs();

    const byteString = atob(imageSrc.split(",")[1]);
    const mimeString = imageSrc.split(",")[0].split(":")[1].split(";")[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([ab], { type: mimeString });
    const file = new File([blob], "webcam.jpg", { type: mimeString });

    setSelectedFile(file);
    setPreviewUrl(imageSrc);
  };

  const analyzeCameraFrame = async (imageSrc) => {
    if (!imageSrc) return null;
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const targetW = 256;
        const targetH = 256;
        const canvas = document.createElement("canvas");
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0, targetW, targetH);
        const { data } = ctx.getImageData(0, 0, targetW, targetH);

        const gray = new Float32Array(targetW * targetH);
        let sum = 0;
        for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
          const g = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
          gray[p] = g;
          sum += g;
        }
        const mean = sum / gray.length;
        let varAcc = 0;
        for (let i = 0; i < gray.length; i += 1) {
          const d = gray[i] - mean;
          varAcc += d * d;
        }
        const std = Math.sqrt(varAcc / gray.length);

        // Laplacian variance (blur proxy)
        let lapSum = 0;
        let lapSq = 0;
        let n = 0;
        for (let y = 1; y < targetH - 1; y += 1) {
          for (let x = 1; x < targetW - 1; x += 1) {
            const c = gray[y * targetW + x];
            const l = gray[y * targetW + (x - 1)];
            const r = gray[y * targetW + (x + 1)];
            const u = gray[(y - 1) * targetW + x];
            const d = gray[(y + 1) * targetW + x];
            const lap = (l + r + u + d - 4 * c);
            lapSum += lap;
            lapSq += lap * lap;
            n += 1;
          }
        }
        const lapMean = lapSum / Math.max(1, n);
        const lapVar = lapSq / Math.max(1, n) - lapMean * lapMean;

        resolve({
          brightness: mean,
          contrast: std,
          blurVar: Math.max(0, lapVar),
        });
      };
      img.onerror = () => resolve(null);
      img.src = imageSrc;
    });
  };

  useEffect(() => {
    if (mode !== "camera") return undefined;
    let cancelled = false;

    const tick = async () => {
      if (!webcamRef.current) return;
      const shot = webcamRef.current.getScreenshot();
      if (!shot) return;
      const m = await analyzeCameraFrame(shot);
      if (!m || cancelled) return;

      let lighting = "good";
      let lightingLevel = "ok";
      if (m.brightness < 0.16) {
        lighting = "very dark";
        lightingLevel = "bad";
      } else if (m.brightness < 0.24) {
        lighting = "dark";
        lightingLevel = "warn";
      } else if (m.brightness > 0.92) {
        lighting = "very bright";
        lightingLevel = "bad";
      } else if (m.brightness > 0.84) {
        lighting = "bright";
        lightingLevel = "warn";
      }

      let contrast = "good";
      let contrastLevel = "ok";
      if (m.contrast < 0.040) {
        contrast = "very low";
        contrastLevel = "bad";
      } else if (m.contrast < 0.055) {
        contrast = "low";
        contrastLevel = "warn";
      }

      let sharpness = "good";
      let sharpnessLevel = "ok";
      if (m.blurVar < 0.0012) {
        sharpness = "very blurry";
        sharpnessLevel = "bad";
      } else if (m.blurVar < 0.0020) {
        sharpness = "blurry";
        sharpnessLevel = "warn";
      }

      let tip = "Great conditions. Keep your face centered inside the oval.";
      if (lightingLevel === "bad") tip = "Do not capture yet: fix lighting first.";
      else if (sharpnessLevel === "bad") tip = "Do not capture yet: image is too blurry.";
      else if (contrastLevel === "bad") tip = "Do not capture yet: scene contrast is too low.";
      else if (lightingLevel === "warn") tip = "Improve front lighting for more reliable analysis.";
      else if (sharpnessLevel === "warn") tip = "Hold still and let the camera focus before capture.";
      else if (contrastLevel === "warn") tip = "Use more even lighting to improve skin detail contrast.";

      setCameraGuide({
        ready: true,
        lighting,
        lightingLevel,
        sharpness,
        sharpnessLevel,
        contrast,
        contrastLevel,
        tip,
      });
    };

    tick();
    const id = setInterval(tick, 1400);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [mode]);

  const prettyQualityReason = (r) => {
    if (!r) return "Unknown issue";
    if (r.startsWith("too_dark")) return "Too dark (increase lighting)";
    if (r.startsWith("too_bright")) return "Too bright (avoid harsh light)";
    if (r.startsWith("blurry")) return "Blurry (hold still / focus)";
    if (r.startsWith("low_contrast")) return "Low contrast (even lighting needed)";
    if (r.startsWith("low_resolution")) return "Low resolution (use higher quality image)";
    if (r.startsWith("face_too_small")) return "Face too small in frame (move closer)";
    if (r.startsWith("face_off_center")) return "Face is off-center (align inside guide)";
    if (r.startsWith("uneven_lighting")) return "Uneven lighting / side shadow (use front light)";
    if (r.startsWith("landmarks_unstable")) return "Face landmarks unstable (face camera directly)";
    if (r.startsWith("eye_region_occluded")) return "Eye region occluded (remove glasses/hair obstruction)";
    if (r.startsWith("face_partially_out_of_frame")) return "Face partly out of frame (re-center in camera)";
    if (r.startsWith("skin_visible_too_low")) return "Too much occlusion over skin (remove mask/hand/hair)";
    return r;
  };

  const buildQualityMessage = (quality, retakeGuidance = []) => {
    const reasons = (quality?.reasons || []).map(prettyQualityReason);
    const tips = (retakeGuidance || []).filter(Boolean);
    if (!reasons.length && !tips.length) return "Image quality too low. Please retake.";
    const reasonText = reasons.length ? `Retake required: ${reasons.join(" • ")}` : "Retake required.";
    const tipsText = tips.length ? ` Tips: ${tips.join(" • ")}` : "";
    return `${reasonText}${tipsText}`;
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select or capture an image first.");
      return;
    }

    setLoading(true);
    resetOutputs();

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const stored =
        localStorage.getItem("auraiUser") ||
        sessionStorage.getItem("auraiUser");

      if (stored) {
        const user = JSON.parse(stored);
        if (user?.id) {
          formData.append("user_id", String(user.id));
        }
      }

      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;

      // Phase 1 quality gate (and also face-not-found messages)
      if (data?.ok === false) {
        if (data?.quality?.passed === false) {
          setError(buildQualityMessage(data.quality, data.retake_guidance));
        } else if (data?.message) {
          setError(String(data.message));
        } else {
          setError("Scan could not be completed. Please retake the photo.");
        }
        return; // do NOT save history
      }

      const newResults = data.results || null;
      const newSkinType = data.skin_type || null;
      const newRoutine = data.routine || null; // filtered routine
      const newRoutineRaw = data.routine_raw || null;
      const newSafety = data.safety || null;
      const newOutcome = data.outcome || null;
      const newRisk = data.risk || null;

      // Explainability/full overlay keys (new)
      const imagePath = data.image_path || null; // backward compat
      const imagePathFull = data.image_path_full || null;
      const imagePathFace = data.image_path_face || null;
      const faceBbox = data.face_bbox || data.face?.bbox || null;

      setResults(newResults);
      setSkinType(newSkinType);

      // Save scan history (works for remember-me and non-remember)
      try {
        const stored2 =
          localStorage.getItem("auraiUser") ||
          sessionStorage.getItem("auraiUser");

        if (stored2) {
          const user = JSON.parse(stored2);
          if (user?.id) {
            await axios.post(`${API_BASE}/scans`, {
              user_id: user.id,
              results: {
                results: newResults,
                skin_type: newSkinType,
                routine: newRoutine,
                routine_raw: newRoutineRaw,
                safety: newSafety,
                outcome: newOutcome,
                risk: newRisk,

                // explainability + image persistence
                image_path: imagePath,                 // optional
                image_path_full: imagePathFull,         // ✅ for /explain full overlay
                image_path_face: imagePathFace,         // ✅ for /explain full overlay
                face_bbox: faceBbox,                    // ✅ for /explain full overlay
              },
            });
          }
        }
      } catch (saveErr) {
        console.error("Failed to save scan history:", saveErr);
      }
    } catch (err) {
      console.error(err);
      setError("Something went wrong while contacting the server.");
    } finally {
      setLoading(false);
    }
  };

  const prettyName = {
    acne: "Acne",
    bags: "Under-eye bags",
    blackheads: "Blackheads",
    hyperpigmentation: "Hyperpigmentation",
    redness: "Redness",
  };

  return (
    <div className="app-root">
      <AppHeader />

      <main className="app-main scan-main">
        {/* LEFT: input controls */}
        <section className="card scan-card scan-card-input">
          <div className="scan-card-head">
            <span className="scan-step-pill">Step 1</span>
            <h2>Choose input method</h2>
          </div>

          <div className="mode-toggle">
            <button
              className={mode === "upload" ? "mode-btn active" : "mode-btn"}
              onClick={() => {
                setMode("upload");
                setSelectedFile(null);
                setPreviewUrl(null);
                resetOutputs();
              }}
            >
              Upload photo
            </button>

            <button
              className={mode === "camera" ? "mode-btn active" : "mode-btn"}
              onClick={() => {
                setMode("camera");
                setSelectedFile(null);
                setPreviewUrl(null);
                resetOutputs();
              }}
            >
              Live camera
            </button>
          </div>

          {mode === "upload" && (
            <div className="upload-area">
              <label className="upload-label">
                <span className="upload-icon">↑</span>
                <span>Drop an image here or click to browse</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
              </label>
              <p className="hint">
                Use a clear, front-facing photo in soft, even lighting.
              </p>
            </div>
          )}

          {mode === "camera" && (
            <div className="camera-area">
              <div className="camera-frame">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  mirrored={true}
                  screenshotFormat="image/jpeg"
                  screenshotQuality={0.92}
                  forceScreenshotSourceSize={true}
                  videoConstraints={{
                    facingMode: "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                  }}
                  className="webcam-preview"
                />

                {/* Overlay guide */}
                <div className="face-overlay" aria-hidden="true">
                  <div className="overlay-status">
                    <div className="overlay-status-pills">
                      <span className={`camera-pill ${cameraGuide.lightingLevel}`}>
                        Light: {cameraGuide.lighting}
                      </span>
                      <span className={`camera-pill ${cameraGuide.sharpnessLevel}`}>
                        Focus: {cameraGuide.sharpness}
                      </span>
                      <span className={`camera-pill ${cameraGuide.contrastLevel}`}>
                        Contrast: {cameraGuide.contrast}
                      </span>
                    </div>
                  </div>
                  <div className="face-oval" />
                  <div className="overlay-hint">
                    {cameraGuide.tip}
                  </div>
                </div>
              </div>

              <button className="primary-btn" onClick={captureFromWebcam}>
                Capture snapshot
              </button>
              <p className="hint">
                Keep your face centered, eyes open, even lighting.
              </p>
            </div>
          )}
        </section>

        {/* MIDDLE: preview */}
        <section className="card scan-card scan-card-preview">
          <div className="scan-card-head">
            <span className="scan-step-pill">Step 2</span>
            <h2>Preview</h2>
          </div>

          {previewUrl ? (
            <div className="scan-preview-frame">
              <img src={previewUrl} alt="preview" className="image-preview" />
            </div>
          ) : (
            <div className="empty-preview">No image selected yet.</div>
          )}

          <button
            className="primary-btn analyze-btn"
            onClick={handleAnalyze}
            disabled={loading || !previewUrl}
          >
            {loading ? "Analyzing..." : "Analyze skin"}
          </button>

          {error && <p className="error-msg">{error}</p>}
        </section>

        {/* RIGHT: results */}
        <section className="card scan-card scan-card-results">
          <div className="scan-card-head">
            <span className="scan-step-pill">Step 3</span>
            <h2>Results</h2>
          </div>

          {skinType && (
            <div className="skin-type-card">
              <div className="skin-type-row">
                <span className="skin-type-label">Skin type</span>
                <span className="skin-type-value">
                  {String(skinType.skin_type || "").toUpperCase()}
                </span>
              </div>
              <div className="skin-type-confidence">
                Confidence: {((skinType.confidence || 0) * 100).toFixed(1)}%
              </div>
            </div>
          )}

          {!results && <p className="hint">Your AI analysis will appear here.</p>}

          {results && (
            <div className="scan-results-wrap">
              <div className="results-grid">
                {Object.entries(results).map(([key, value]) => {
                  const prob = value?.probability || 0;
                  const present = prob >= UI_DETECT_THRESHOLD;
                  const percent = (prob * 100).toFixed(1);

                  return (
                    <div
                      key={key}
                      className={
                        "result-card " + (present ? "result-present" : "result-absent")
                      }
                    >
                      <div className="result-header">
                        <span className="result-title">
                          {prettyName[key] || key}
                        </span>
                        <span className="result-tag">
                          {present ? "Detected" : "Low visibility"}
                        </span>
                      </div>

                      <div className="result-bar-wrapper">
                        <div className="result-bar-bg">
                          <div
                            className="result-bar-fill"
                            style={{
                              width: `${Math.min(Math.max(prob * 100, 5), 100)}%`,
                            }}
                          />
                        </div>
                        <span className="result-percent">{percent}%</span>
                      </div>

                      <p className="result-note">
                        {present
                          ? "This condition is visible. Consider a focused skincare routine."
                          : "This condition is not strongly visible in this photo."}
                      </p>
                    </div>
                  );
                })}
              </div>
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

export default Scan;
