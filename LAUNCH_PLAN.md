# Launch Plan & Feature Checklist

## Week 1: Input + Guardrails

- [ ] **Lock capture standards in Scan.jsx**
  - Fixed pose guidance
  - Lighting hints
  - No heavy angle
- [ ] **Strengthen backend quality gate in main.py (/predict)**
  - Stricter blur/shadow/face-size thresholds
- [ ] **Add explicit retake reasons in UI**
  - "too dark", "off-center", "blurry"
- [ ] **Save quality metrics per scan (DB JSON)**
  - Audit bad inputs later

## Week 2: Model Reliability

- [ ] **Keep hybrid stack**
  - YOLO for localized (acne, blackheads, bags)
  - Classifier/ROI for diffuse (redness, hyperpigmentation)
- [ ] **Tune thresholds from calibration output**
  - Load per_class_thresholds_by_tone_tuned.json
  - Add safe floor for thresholds
- [ ] **Add uncertainty gating in prediction response**
  - "low visibility / uncertain" band
- [ ] **Add confidence policy in UI**
  - Never show "detected" unless confidence clears display threshold

## Week 3: Explainability + Outcome Preview Quality

- [ ] **Finalize explain modes**
  - Boxes for localized
  - Soft overlays for diffuse
- [ ] **Improve outcome simulation**
  - Subtle / Balanced / Strong toggle
  - Preserve natural skin texture
- [ ] **Add skin-only mask before explain/simulation blend**
  - Avoid hair/lips influence
- [ ] **Put disclaimer under previews**
  - "visual projection, not medical guarantee"

## Week 4: Regression + Launch Readiness

- [ ] **Expand regression suite to 100+ curated cases**
  - Shadows, facial hair, makeup, tones/lighting, edge poses
- [ ] **Automate regression check before every merge**
  - Fail if pass rate < target
- [ ] **Add release dashboard JSON**
  - Per-label precision/recall trend
  - False-positive counts by condition
  - Calibration summary by skin tone
- [ ] **Freeze launch candidate model + thresholds + config**
  - Version tags

## Launch Targets

- Regression pass rate: >= 95%
- Diffuse-label false positives (shadow-driven): cut by 50%+
- Stable UX: no contradictory outputs
- All scans have quality metadata + explainability mode returned
