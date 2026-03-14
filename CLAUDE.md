# CLAUDE.md — AURAI Skincare AI Project

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project context

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Ask yourself: would a staff engineer approve this?
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Project: AURAI — AI Skincare Analysis App

### What It Does
End-to-end skin analysis pipeline:
1. Face capture (upload or camera)
2. Image quality gate (blur, brightness, centering, shadow)
3. Face parsing via BiSeNet (skin-only mask)
4. Multilabel concern detection: acne, bags, blackheads, hyperpigmentation, redness
5. Skin type + skin tone classification
6. YOLO-based localized lesion detection (cross-checks CNN output)
7. Grad-CAM explainability overlays
8. Personalized skincare routine recommendation (from Excel formula templates)
9. Safety filtering (allergen removal)
10. Outcome simulation (before/after visual preview)
11. Scan history stored in MySQL

### How to Run

#### Backend (FastAPI)
```bash
# From project root (NOT from inside /api)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (React)
```bash
cd frontend
npm install
npm start   # runs on http://localhost:3000
```

#### Release Check
```bash
bash scripts/release_check.sh
```

### Environment Variables (Required — NOT in source)
Create `.env` in project root:
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=<your_password>
MYSQL_DB=aurai
```
The file `.env` is in `.gitignore`. Never hardcode credentials.

### Model Files (in `models/`)
| File | Purpose | Size |
|------|---------|------|
| `best_multilabel.pt` | Main ResNet18 — 5 skin concerns | ~45MB |
| `skin_type_resnet18.pt` | Skin type classifier (5 classes) | ~45MB |
| `skin_tone_resnet18.pt` | Skin tone classifier (3 classes) | ~45MB |
| `bisenet_face_parsing.pth` | Face segmentation | 51MB |
| `yolo_skin_best.pt` | YOLO lesion detector | varies |
| `per_class_thresholds_by_tone_tuned.json` | Operative thresholds (tone-aware) | — |
| `classes.json` | Label list: acne, bags, blackheads, hyperpigmentation, redness | — |

### Key Architecture Decisions
- **ResNet18** for multilabel (currently; EfficientNet-B3 upgrade planned)
- **BiSeNet** for face parsing (19-class segmentation, skin labels = {1, 10})
- **MediaPipe BlazeFace** for face detection (`model_selection=1` preferred)
- **YOLO** for localized lesion detection — acts as cross-check on CNN
- **Tone-aware thresholds**: loaded from `per_class_thresholds_by_tone_tuned.json`, require `tone_conf >= 0.8`
- **Uncertainty gating**: suppresses weak positives in `_apply_uncertainty_gating()` in `main.py`
- **YOLO fusion**: when YOLO finds nothing, acne/blackheads/bags prob multiplied by 0.35

### Critical Known Issues (Do Not Regress)
- Quality gate must run on **`face_img_raw`** (raw crop), NOT the oval-masked image
- Shadow score gates at two points: quality gate (`> 0.35`) and hyperpigmentation overlay suppression (`> 0.22`) — both wired ✓
- tone_conf_min default is 0.8 in inference.py ✓
- Threshold gating uses thr_map as single source of truth (no hardcoded floors) ✓
- Rate limiting: 10/min per IP on `/predict`; admins bypass via DB role check ✓
- Passwords use bcrypt (SHA-256 fallback for legacy accounts only) ✓
- CORS restricted to `CORS_ORIGINS` env var (defaults to localhost:3000) ✓

### File Structure
```
project-root/
├── api/                    # FastAPI backend
│   ├── main.py             # Main app + all routes (1862 lines — needs splitting)
│   ├── inference.py        # SkinModel, SkinTypeModel, SkinToneModel
│   ├── quality_gate.py     # Image quality assessment
│   ├── face_crop.py        # MediaPipe face detection + oval mask
│   ├── face_parse.py       # BiSeNet skin mask
│   ├── face_rois.py        # FaceMesh ROI extraction
│   ├── explainability.py   # Grad-CAM overlay generation
│   ├── yolo_detector.py    # YOLO wrapper
│   ├── recommender.py      # Excel formula-based routine engine
│   ├── outcome_risk.py     # Ingredient benefit/risk scoring
│   ├── safety_filter.py    # Allergen filtering
│   └── models/
│       └── bisenet.py      # BiSeNet architecture
├── frontend/               # React SPA
│   └── src/
│       ├── pages/          # Scan, History, Login, Register, Admin*, etc.
│       └── components/     # AppHeader, CameraInput, UploadBox, ResultCard
├── models/                 # Model weights + config JSON files
├── scripts/                # Training + eval + release gate scripts
├── src/                    # model.py (build_resnet18_multilabel), data.py
├── data/formulas/          # Excel formula files per skin type/step
├── tasks/                  # todo.md + lessons.md (create if missing)
└── CLAUDE.md               # This file
```

### Planned Improvements (Priority Order)
1. Security: bcrypt passwords, JWT auth, .env credentials, rate limiting
2. Quality gate: wire shadow score gate, add face angle check
3. Model: upgrade main classifier to EfficientNet-B3
4. Inference: batch ROI forward passes, add TTA, add model warm-up
5. Backend: DB connection pooling, split main.py into route modules
6. Frontend: env-based API URL, sessionStorage for auth tokens
7. Explainability: use YOLO boxes for localized concerns instead of Grad-CAM

### DB Schema (MySQL — `aurai` database)
- `users`: id, name, email, password_hash, role, phone, age, address, allergies
- `scans`: id, user_id, created_at, results_json
- `scan_annotations`: id, scan_id, user_id, image_kind, annotations_json, notes, updated_at

### Frontend API Base
Currently hardcoded to `http://127.0.0.1:8000` in every page file.
Must be moved to `process.env.REACT_APP_API_BASE` before deployment.
