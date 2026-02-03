# Unused/Deprecated Files in AURAI Project

## UNUSED SCRIPTS (Can be removed)

### 1. **`frontend/src/pages/Orders.jsx`** ❌
- **Status**: COMPLETELY UNUSED
- **Reason**: Never imported in `App.js` or referenced anywhere
- **Recommendation**: DELETE - no functionality needed for skin analysis app
- **Route**: `/orders` (not in App.js routing)

### 2. **`scripts/Dashboard.py`** ⚠️  DEPRECATED
- **Status**: Streamlit dashboard (superseded by React frontend)
- **Reason**: Frontend has replaced all dashboard functionality
- **Current Use**: None (React Scan.jsx handles all predictions)
- **Recommendation**: DELETE - React frontend is the modern interface
- **Dependencies**: Would need Streamlit installed (not in requirements)

### 3. **`scripts/dataset_skin.py`** ⚠️ LIKELY UNUSED
- **Status**: Generic dataset class for older pipeline
- **Reason**: `train_multilabel.py` uses `MultiLabelCSVDataset`, not `SkinDefectsDataset`
- **Used By**: Nothing in current codebase
- **Recommendation**: DELETE - Replaced by `MultiLabelCSVDataset` in train_multilabel.py
- **Purpose**: Basic CSV→PIL loader (superseded by more robust version)

### 4. **`scripts/metadataold/` (entire folder)** ⚠️ DEPRECATED
- **Files in folder**:
  - `make_metadata.py` - OLD version
  - `make_metadata_unid_csv.py` - OLD version
  - `merge_metadata.py` - OLD version
  - `train_skin.py` - OLD version
- **Status**: Replaced by `make_metadata_all_multilabel.py`
- **Reason**: Active script is in `/scripts/` folder, not in `/metadataold/`
- **Recommendation**: DELETE - Archival only, use `make_metadata_all_multilabel.py` instead

---

## ACTIVE SCRIPTS (Keep these)

### Production/Inference
- ✅ `api/main.py` - FastAPI server
- ✅ `api/inference.py` - Model loading & prediction
- ✅ `api/quality_gate.py` - Image quality checks
- ✅ `api/face_crop.py` - Face detection & cropping
- ✅ `api/recommender.py` - Skincare routine generation
- ✅ `api/safety_filter.py` - Allergy filtering
- ✅ `api/outcome_risk.py` - Benefit/risk scoring
- ✅ `api/explainability.py` - GradCAM heatmaps

### Training
- ✅ `scripts/train_multilabel.py` - Train 5-label skin concerns
- ✅ `scripts/train_skin_type.py` - Train 5-class skin type

### Evaluation
- ✅ `scripts/eval_skin.py` - Evaluate multilabel model (classification reports, F1)
- ✅ `scripts/eval_skin_type.py` - Evaluate skin type model (confusion matrix)

### Data Preparation
- ✅ `scripts/make_metadata_all_multilabel.py` - Build training CSVs with stratified splits
- ✅ `scripts/find_thresholds.py` - Optimize per-class prediction thresholds
- ✅ `scripts/inspect_formulas.py` - Validate skincare formula Excel data

### Frontend
- ✅ `frontend/src/pages/Home.jsx`
- ✅ `frontend/src/pages/Login.jsx`
- ✅ `frontend/src/pages/Register.jsx`
- ✅ `frontend/src/pages/Scan.jsx` - Main prediction interface
- ✅ `frontend/src/pages/History.jsx` - View past scans
- ✅ `frontend/src/pages/Account.jsx` - User profile
- ✅ `frontend/src/pages/Profile.jsx` - Edit allergies
- ✅ `frontend/src/components/*` - All components

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| **Unused Files** | 2 | DELETE |
| **Deprecated Folders** | 1 | DELETE (`metadataold/`) |
| **Unused Datasets** | 1 | DELETE |
| **Active Production** | 8 modules | KEEP |
| **Active Training** | 2 scripts | KEEP |
| **Active Evaluation** | 2 scripts | KEEP |
| **Active Data Prep** | 3 scripts | KEEP |
| **Active Frontend** | 12 files | KEEP |

**Total files to remove**: 5
**Total files to keep**: ~30 (production + training + eval + data + frontend)

---

## Cleanup Checklist

```bash
# Remove unused files
rm frontend/src/pages/Orders.jsx
rm scripts/Dashboard.py
rm scripts/dataset_skin.py
rm -rf scripts/metadataold/

# Optional: Archive to archive/ folder instead of deleting
mv frontend/src/pages/Orders.jsx archive/
mv scripts/Dashboard.py archive/
mv scripts/dataset_skin.py archive/
mv scripts/metadataold/ archive/
```
