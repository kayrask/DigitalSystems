# AURAI System Architecture & UML Diagrams

## Overview
AURAI is an AI-powered facial skin analysis platform with:
- **Backend API**: FastAPI + PyTorch for skin condition detection, skin type classification, personalized routine recommendation
- **Frontend**: React SPA for UI
- **Database**: MySQL for user management and scan history
- **ML Pipeline**: Training scripts, metadata processing, model evaluation

---

## 1. CLASS DIAGRAM - Backend API & Inference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKEND API ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│     SkinModel            │
├──────────────────────────┤
│ - device: str            │
│ - model: ResNet18        │
│ - thresholds: Dict       │
│ - transform: Compose     │
├──────────────────────────┤
│ + __init()               │
│ + _build_model()         │
│ + predict(image): Dict   │
└──────────────────────────┘
         △
         │ uses
         │
    ┌────┴─────────────────────────────────────┐
    │                                           │
    
┌──────────────────────────┐    ┌─────────────────────────┐
│   SkinTypeModel          │    │   FastAPI App           │
├──────────────────────────┤    ├─────────────────────────┤
│ - device: torch.device   │    │ - models: Dict          │
│ - model: ResNet18        │    │ - connection: MySQL     │
│ - classes: List[str]     │    │ - CORS: CORSMiddleware  │
│ - tf: Compose            │    ├─────────────────────────┤
├──────────────────────────┤    │ + GET /                 │
│ + __init()               │    │ + POST /predict         │
│ + predict(image): Dict   │    │ + POST /register        │
└──────────────────────────┘    │ + POST /login           │
                                │ + POST /scans           │
                                │ + GET /scans            │
                                └─────────────────────────┘
                                         △
                                         │ calls
                                         │
        ┌────────────────┬───────────────┼─────────────────┬────────────────┐
        │                │               │                 │                │
        
┌──────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────┐
│  recommend_routine()     │ │ assess_image_quality()  │ │ crop_face()      │
├──────────────────────────┤ ├─────────────────────────┤ ├──────────────────┤
│ + concern_probs: Dict    │ │ + img: PILImage         │ │ + image: PILImage│
│ + skin_type: str         │ │ + cfg: QualityConfig    │ │ + margin: float  │
├──────────────────────────┤ ├─────────────────────────┤ ├──────────────────┤
│ - ING_DB: Dict           │ │ + passed: bool          │ │ + face_found: bool
│ - TWEAKABLE: Dict        │ │ + reasons: List[str]    │ │ + bbox: Dict     │
├──────────────────────────┤ │ + metrics: Dict         │ ├──────────────────┤
│ + parse_range()          │ │ + thresholds: Dict      │ │ + _pil_to_gray()│
│ + personalize_formula()  │ ├─────────────────────────┤ │ + apply_oval()  │
│ + _score_ing()           │ │ - brightness: float     │ └──────────────────┘
│ - _load_ingredient_db()  │ │ - contrast: float       │
└──────────────────────────┘ │ - blur_var: float       │
        △                     └─────────────────────────┘
        │ uses
        │
┌────────────────────────────────────────────────┐
│         QualityConfig (dataclass)              │
├────────────────────────────────────────────────┤
│ - min_short_side: int = 320                    │
│ - brightness_min: float = 0.20                 │
│ - brightness_max: float = 0.85                 │
│ - contrast_min: float = 0.05                   │
│ - blur_var_min: float = 0.0015                 │
└────────────────────────────────────────────────┘


┌──────────────────────────────────────────────┐
│  filter_routine_for_user()                   │
├──────────────────────────────────────────────┤
│ + routine: Dict                              │
│ + allergies_text: str | None                 │
├──────────────────────────────────────────────┤
│ - LOCKED_KEYWORDS: List[str]                 │
│ - ALLERGY_ALIASES: Dict                      │
│ - SafetyConfig: dataclass                    │
├──────────────────────────────────────────────┤
│ + _tokenize_allergies()                      │
│ + _is_locked_inci()                          │
│ + _find_water_index()                        │
│ + _rebalance_to_100()                        │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  compute_outcome_and_risk()                  │
├──────────────────────────────────────────────┤
│ + routine: Dict                              │
│ + concern_probs: Dict                        │
│ + skin_type: str                             │
├──────────────────────────────────────────────┤
│ - ACTIVE_BENEFIT: Dict                       │
│ - IRRITANT_FLAGS: List[Tuple]                │
│ - STRONG_ACTIVES: List[Tuple]                │
│ - Phase3Config: dataclass                    │
├──────────────────────────────────────────────┤
│ + _extract_inci_tokens()                     │
│ + score_benefit()                            │
│ + score_risk()                               │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  gradcam_overlay_base64()                    │
├──────────────────────────────────────────────┤
│ + skin_model: SkinModel                      │
│ + image: PILImage                            │
│ + label_idx: int                             │
├──────────────────────────────────────────────┤
│ + _make_red_overlay_rgba()                   │
│ + returns: base64 PNG string                 │
└──────────────────────────────────────────────┘
```

---

## 2. CLASS DIAGRAM - Data & Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATA & TRAINING ARCHITECTURE                           │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│         SkinDataset (torch.Dataset)          │
├──────────────────────────────────────────────┤
│ - df: pd.DataFrame                           │
│ - images_root: str                           │
│ - classes: List[str]                         │
│ - class_to_idx: Dict[str, int]               │
│ - tfm: transforms.Compose                    │
├──────────────────────────────────────────────┤
│ + __len__(): int                             │
│ + __getitem__(idx): Tuple[Tensor, Tensor]    │
│   returns (image_tensor, multilabel_tensor)  │
└──────────────────────────────────────────────┘
         △                                    △
         │ used by                           │ uses
         │                           ┌───────┴──────────┐
         │                           │                  │
    ┌────┴─────────────────────┐    │                  │
    │                          │    │                  │
    
┌─────────────────────────────────────────┐   ┌─────────────────────────────┐
│  MultiLabelCSVDataset                   │   │  get_transforms()           │
│  (torch.utils.data.Dataset)             │   ├─────────────────────────────┤
├─────────────────────────────────────────┤   │ + img_size: int = 224       │
│ - df: pd.DataFrame                      │   ├─────────────────────────────┤
│ - split: str (train|val|test)           │   │ returns:                    │
│ - img_col: str = "image_path"           │   │  (train_tf, eval_tf)        │
├─────────────────────────────────────────┤   │                             │
│ + __len__(): int                        │   │ train_tf:                   │
│ + __getitem__(idx):                     │   │  - RandomResizedCrop(224)   │
│   Tuple[Tensor, multi_hot_labels]       │   │  - RandomHorizontalFlip     │
│                                         │   │  - ColorJitter              │
│ LABELS = [acne, bags, blackheads, ...]  │   │  - RandomPerspective        │
│                                         │   │  - GaussianBlur             │
└─────────────────────────────────────────┘   │  - RandomErasing            │
                                              │  - Normalize                │
                                              │                             │
                                              │ eval_tf:                    │
                                              │  - Resize(224)              │
                                              │  - ToTensor()               │
                                              │  - Normalize                │
                                              └─────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                         build_resnet18_multilabel()                      │
├──────────────────────────────────────────────────────────────────────────┤
│ Parameters:                                                              │
│  - num_classes: int                                                      │
│  - pretrained: bool = True                                               │
├──────────────────────────────────────────────────────────────────────────┤
│ Returns: ResNet18 with custom FC layer                                   │
│  - model.fc = Linear(512, num_classes) for multilabel                   │
│  - Loss: BCEWithLogitsLoss (via torch.sigmoid)                          │
│  - Activation: Sigmoid (per-label independent predictions)              │
└──────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                   Training Loop (train_multilabel.py)                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each epoch:                                                         │
│    1. Load train_ds with DataLoader(batch_size=16, shuffle=True)        │
│    2. Forward: logits = model(img) [batch_size, 5]                      │
│    3. Loss: BCEWithLogitsLoss(logits, multi_hot_labels)                 │
│    4. Backward + optimizer.step()                                        │
│    5. Eval on val_ds:                                                    │
│       - Compute micro F1 score                                           │
│       - Save best checkpoint if val_loss improves                        │
│    6. Test on test_ds at end                                             │
│                                                                          │
│  Output: models/best_multilabel.pt                                      │
│          models/per_class_thresholds.json (from find_thresholds.py)    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                Training Loop (train_skin_type.py)                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Load from data_dir/train, data_dir/validate, data_dir/test          │
│     (using torchvision.datasets.ImageFolder)                            │
│  2. Classes: Combination, Dry, Normal, Oily, Sensitive                  │
│  3. Model: ResNet18 with custom FC                                       │
│     - model.fc = Linear(512, num_classes=5)                             │
│  4. Loss: CrossEntropyLoss (multi-class classification)                 │
│  5. Activation: Softmax (mutually exclusive prediction)                 │
│  6. Best checkpoint saved to models/skin_type_resnet18.pt               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. STATE DIAGRAM - User & Prediction Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│           USER LIFECYCLE STATE DIAGRAM (Frontend + API)                 │
└─────────────────────────────────────────────────────────────────────────┘

                              START
                                │
                                ▼
                        ┌──────────────────┐
                        │  Home (Public)   │
                        │  No user in      │
                        │  localStorage    │
                        └────┬─────────────┘
                             │
                ┌────────────┴────────────┐
                │                        │
                ▼                        ▼
        ┌─────────────────┐      ┌──────────────┐
        │  Login Page     │      │ Register Page│
        │  POST /login    │      │ POST /register
        └────┬────────────┘      └──────┬───────┘
             │                          │
             │ success                  │ success
             │                          │
             └──────────────┬───────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │ Store user in localStorage    │
            │ or sessionStorage             │
            │ (depends on "Remember me")    │
            └───────────────┬───────────────┘
                            │
                            ▼
                   ┌─────────────────────┐
                   │ Home (Authenticated)│
                   │ Show AppHeader      │
                   │ Navigate to /scan   │
                   └──────────┬──────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
            ┌────────┐  ┌──────────┐  ┌─────────┐
            │ Scan   │  │ History  │  │ Account │
            │ Page   │  │ Page     │  │ Page    │
            │ (New)  │  │ (Browse) │  │ (View)  │
            └────┬───┘  └──────────┘  └─────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │   Upload/Capture Image         │
    │   POST /predict with image     │
    └────────┬───────────────────────┘
             │
             ▼ Processing
    ┌──────────────────────────────────────────────┐
    │  1. Quality Check (assess_image_quality)    │
    │     → passed? yes/no + reasons              │
    └────┬───────────────────────────────────────┘
         │
         ├─ NO (low quality)
         │  └─→ Return error + suggestions
         │
         └─ YES
            ▼
    ┌──────────────────────────────────────────────┐
    │  2. Face Detection & Crop (crop_face)       │
    │     → face_found? yes/no + bbox + confidence│
    └────┬───────────────────────────────────────┘
         │
         ├─ NO face
         │  └─→ Return error
         │
         └─ YES
            ▼
    ┌──────────────────────────────────────────────┐
    │  3. Apply Oval Mask (apply_oval_mask)       │
    │     → smooth-feathered crop                  │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  4. Skin Type Prediction (SkinTypeModel)    │
    │     → skin_type: str (Normal/Oily/Dry/...)  │
    │     → confidence: float                      │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  5. Skin Concerns (SkinModel)                │
    │     → {acne, bags, blackheads, redness, ...}│
    │     → each: {probability, prediction}        │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  6. Recommend Routine (recommend_routine)   │
    │     → top_concerns: List[str]               │
    │     → suggested_ingredients: List[str]      │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  7. Safety Filter (filter_routine_for_user) │
    │     → match user allergies                   │
    │     → remove unsafe ingredients              │
    │     → rebalance formula                      │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  8. Outcome & Risk (compute_outcome_risk)   │
    │     → benefit_score: float                   │
    │     → risk_score: float                      │
    │     → reasoning: str                         │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────────────────┐
    │  9. Explainability (gradcam_overlay)        │
    │     → heatmap of model attention             │
    │     → base64 PNG for display                 │
    └────┬───────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────┐
    │  10. Save to DB (POST /scans)                 │
    │      results_json → scans table               │
    └────┬─────────────────────────────────────────┘
         │
         ▼
    ┌──────────────────────────────────┐
    │  Display Results                 │
    │  - Skin Type                     │
    │  - Condition Probabilities       │
    │  - Routine Recommendations       │
    │  - Heatmap Overlay               │
    │  - Risk/Benefit Analysis         │
    └──────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────┐
    │  Save to History (scans table)   │
    │  Browse via /history endpoint    │
    └──────────────────────────────────┘
             │
             └──→ Back to Home / Scan Again
```

---

## 4. ENTITY-RELATIONSHIP DIAGRAM (Database)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MYSQL DATABASE SCHEMA                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│          users                   │
├──────────────────────────────────┤
│ PK  id (INT, AUTO_INCREMENT)     │
│ UQ  email (VARCHAR 255)          │
│     name (VARCHAR 255)           │
│     password_hash (VARCHAR 64)   │
│     phone (VARCHAR 50, NULL)     │
│     age (INT, NULL)              │
│     address (TEXT, NULL)         │
│     allergies (TEXT, NULL)       │
└──────────────────────────────────┘
          │
          │ 1-to-Many (ON DELETE CASCADE)
          │
          ▼
┌──────────────────────────────────────────┐
│          scans                           │
├──────────────────────────────────────────┤
│ PK  id (INT, AUTO_INCREMENT)             │
│ FK  user_id (INT) → users.id             │
│     created_at (TIMESTAMP)               │
│     results_json (TEXT)                  │
│     {                                    │
│       "acne": {...},                     │
│       "bags": {...},                     │
│       "redness": {...},                  │
│       "skin_type": "normal",             │
│       "routine": {...}                   │
│     }                                    │
└──────────────────────────────────────────┘


Result JSON Structure:
{
  "results": {
    "acne": { "probability": 0.82, "prediction": 1 },
    "bags": { "probability": 0.12, "prediction": 0 },
    "blackheads": { "probability": 0.45, "prediction": 1 },
    "hyperpigmentation": { "probability": 0.33, "prediction": 0 },
    "redness": { "probability": 0.19, "prediction": 0 }
  },
  "skin_type": {
    "skin_type": "combination",
    "confidence": 0.94,
    "probs": { "combination": 0.94, "dry": 0.02, ... }
  },
  "routine": {
    "skin_type": "combination",
    "top_concerns": ["acne", "blackheads"],
    "suggested_ingredients": ["niacinamide", "salicylic acid", ...],
    "steps": [...]
  },
  "quality": {
    "passed": true,
    "reasons": [],
    "metrics": { ... }
  },
  "explainability": {
    "heatmap_base64": "iVBORw0KGgoAA..."
  }
}
```

---

## 5. SEQUENCE DIAGRAM - Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│            SEQUENCE: /predict Endpoint Flow                              │
└─────────────────────────────────────────────────────────────────────────┘

User            Frontend           FastAPI           ML Models          Database
 │                 │                  │                  │                  │
 │ "Upload image"  │                  │                  │                  │
 ├────────────────>│                  │                  │                  │
 │                 │ POST /predict    │                  │                  │
 │                 │ (multipart form) │                  │                  │
 │                 ├─────────────────>│                  │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [1] read image   │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [2] quality_gate │                  │
 │                 │                  ├─────────────────>│                  │
 │                 │                  │<─────────────────┤                  │
 │                 │                  │  (passed/failed) │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [3] crop_face    │                  │
 │                 │                  ├─────────────────>│                  │
 │                 │                  │<─────────────────┤                  │
 │                 │                  │  (face + bbox)   │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [4] apply_oval   │                  │
 │                 │                  ├─────────────────>│                  │
 │                 │                  │<─────────────────┤                  │
 │                 │                  │  (masked image)  │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [5] SkinTypeModel│                  │
 │                 │                  ├─────────────────>│                  │
 │                 │                  │<─────────────────┤                  │
 │                 │                  │  (skin_type)     │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [6] SkinModel    │                  │
 │                 │                  ├─────────────────>│                  │
 │                 │                  │<─────────────────┤                  │
 │                 │                  │  (concern probs) │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [7] recommend    │                  │
 │                 │                  │  _routine        │                  │
 │                 │                  │  (loads Excel)   │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [8] filter_      │                  │
 │                 │                  │  routine_for_user│                  │
 │                 │                  │  (user allergies)│                  │
 │                 │                  │                  │                  │
 │                 │                  │ [9] compute_     │                  │
 │                 │                  │  outcome_risk    │                  │
 │                 │                  │                  │                  │
 │                 │                  │ [10] gradcam     │                  │
 │                 │                  │  (explainability)│                  │
 │                 │                  │                  │                  │
 │                 │                  │ [11] POST /scans │                  │
 │                 │                  ├───────────────────────────────────> │
 │                 │                  │                  │        (INSERT)  │
 │                 │                  │<───────────────────────────────────┤
 │                 │                  │                  │      (scan_id)   │
 │                 │                  │                  │                  │
 │                 │  JSON response   │                  │                  │
 │                 │<─────────────────┤                  │                  │
 │<────────────────┤                  │                  │                  │
 │ Display results │                  │                  │                  │
 │                 │                  │                  │                  │

Total time: ~2-5 seconds (depends on model size & image resolution)
```

---

## 6. SYSTEM ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      HIGH-LEVEL SYSTEM DESIGN                            │
└─────────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │   React Frontend │
                          │   (SPA)          │
                          │                  │
                          │  Pages:          │
                          │ - Home           │
                          │ - Login/Register │
                          │ - Scan (upload)  │
                          │ - History        │
                          │ - Account        │
                          │ - Profile        │
                          └────────┬─────────┘
                                   │
                    HTTP / REST API │ JSON
                                   │
                    ┌──────────────▼─────────────┐
                    │     FastAPI Backend        │
                    │                            │
                    │  Endpoints:                │
                    │  - /predict (multipart)    │
                    │  - /register, /login       │
                    │  - /scans (GET/POST)       │
                    │  - /account                │
                    │  - /profile                │
                    │                            │
                    │  Pipeline:                 │
                    │  ├─ quality_gate           │
                    │  ├─ face_crop              │
                    │  ├─ SkinModel (concerns)   │
                    │  ├─ SkinTypeModel (type)   │
                    │  ├─ recommender            │
                    │  ├─ safety_filter          │
                    │  ├─ outcome_risk           │
                    │  └─ explainability         │
                    └──────────┬──────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌─────────────────┐           ┌────────────────┐
        │  PyTorch Models │           │  MySQL Database│
        │                 │           │                │
        │ Models:         │           │  Tables:       │
        │ - best_multilab│           │  - users       │
        │   el.pt         │           │  - scans       │
        │ - skin_type_    │           │                │
        │   resnet18.pt   │           │  Connection:   │
        │                 │           │  localhost:3306│
        │ Thresholds:     │           │                │
        │ - per_class_    │           │  Driver:       │
        │   thresholds.   │           │  mysql-connector
        │   json          │           │                │
        └─────────────────┘           └────────────────┘


Data Flow Layers:

┌─────────────────────────────────────────────────────────────────┐
│ INPUT LAYER: User uploads image                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ QUALITY LAYER: Brightness, contrast, blur checks                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING LAYER: Face detection, cropping, masking           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE LAYER: Dual model prediction                          │
│  - SkinTypeModel → {normal, oily, dry, sensitive, combination}  │
│  - SkinModel → {acne, bags, blackheads, redness, hyperpig}      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RECOMMENDATION LAYER: Routine generation + personalization      │
│  - Excel formula loading                                         │
│  - Ingredient scoring                                            │
│  - Formula tweaking based on severity                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ SAFETY LAYER: Allergy filtering + ingredient removal            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ANALYSIS LAYER: Benefit/risk scoring + explainability           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT LAYER: Return JSON + save to database                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. COMPONENT DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   COMPONENT DEPENDENCIES                                 │
└─────────────────────────────────────────────────────────────────────────┘

                             api/main.py (FastAPI App)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
   api/inference.py         api/recommender.py         api/quality_gate.py
   ├─ SkinModel             ├─ recommend_routine()      ├─ assess_image_quality()
   └─ SkinTypeModel         ├─ personalize_formula()    └─ QualityConfig
                            └─ parse_range()
        │                          │                          │
        │                          ▼                          │
        │              api/safety_filter.py                   │
        │              ├─ filter_routine_for_user()           │
        │              └─ SafetyConfig                        │
        │                          │                          │
        └──────────────┬───────────┼──────────────┬───────────┘
                       │           │              │
                       ▼           ▼              ▼
                  api/face_crop.py
                  ├─ crop_face()
                  └─ apply_oval_mask()
                       │
        ┌──────────────┴───────────────────────┐
        │                                      │
        ▼                                      ▼
   api/outcome_risk.py                   api/explainability.py
   ├─ compute_outcome_risk()             ├─ gradcam_overlay_base64()
   └─ Phase3Config                       └─ _make_red_overlay_rgba()


Training & Data Processing Dependencies:

   scripts/train_multilabel.py
   ├─ src/data.py (SkinDataset)
   ├─ src/model.py (build_resnet18_multilabel)
   └─ MultiLabelCSVDataset

   scripts/train_skin_type.py
   ├─ torchvision.datasets.ImageFolder
   └─ models.resnet18

   scripts/find_thresholds.py
   ├─ models/best_multilabel.pt
   └─ generates: models/per_class_thresholds.json

   scripts/make_metadata_all_multilabel.py
   ├─ data/raw_a/
   ├─ data/raw_b/
   └─ generates: data/processed/all_multilabel.csv

   scripts/inspect_formulas.py
   ├─ data/knowledge/cilt_tipleri.xlsx
   └─ data/formulas/
```

---

## 8. FRONTEND STATE MANAGEMENT

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  REACT STATE & STORAGE ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────┘

                          localStorage / sessionStorage
                                      │
                                      ▼
                          ┌──────────────────────────┐
                          │ User Object (JSON)       │
                          ├──────────────────────────┤
                          │ {                        │
                          │   id: number             │
                          │   name: string           │
                          │   email: string          │
                          │ }                        │
                          │                          │
                          │ Key: "auraiUser"         │
                          └──────────────────────────┘
                                      △
                        ┌─────────────┴─────────────┐
                        │                           │
                  ┌─────┴──────┐           ┌────────┴──────┐
                  │ Remember   │           │ Session Only  │
                  │ Me = true  │           │ Remember Me   │
                  │ → local    │           │ = false       │
                  │   Storage  │           │ → sessionStore│
                  └────────────┘           └───────────────┘
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │  AppHeader Component              │
                    │  (getStoredUser helper)           │
                    │                                   │
                    │  Checks both storages             │
                    │  Renders user pill + logout       │
                    └─────────────────────────────────────┘


Frontend Page States:

Home.jsx:
├─ user: null → show public header
└─ user: obj  → show AppHeader

Login.jsx:
├─ email: string
├─ password: string
├─ rememberMe: boolean
├─ errorMsg: string
└─ handle: POST /login → save user → navigate to /scan

Register.jsx:
├─ name, email, password
├─ address: optional
├─ hasAllergies: "yes"|"no"|null
├─ allergies: string (if yes)
└─ handle: POST /register → navigate to /login

Scan.jsx (implied from routes):
├─ image: File
├─ results: { acne, bags, redness, ... }
└─ handle: POST /predict → display results

History.jsx:
├─ scans: Array[Scan]
├─ openId: number (accordion toggle)
└─ fetch: GET /scans?user_id=X

Account.jsx:
├─ user: { name, email }
└─ actions: [Start new scan]

Profile.jsx:
├─ profile: { address, allergies }
├─ editMode toggles
└─ handle: PUT /profile (if implemented)
```

---

## 9. DEPLOYMENT & CONFIGURATION

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DEPLOYMENT CHECKLIST & CONFIGURATION                       │
└─────────────────────────────────────────────────────────────────────────┘

Backend (API) Setup:
├─ Python Environment
│  ├─ Python 3.9+
│  └─ Virtual env (.venv)
│
├─ Dependencies
│  ├─ pip install -r api/requirements.txt
│  ├─ torch (CPU or GPU)
│  ├─ fastapi, uvicorn
│  ├─ mysql-connector-python
│  ├─ pillow, mediapipe
│  └─ openpyxl (for Excel)
│
├─ Configuration
│  ├─ MYSQL_HOST = "localhost"
│  ├─ MYSQL_USER = "root"
│  ├─ MYSQL_PASSWORD = env var or .env
│  └─ MYSQL_DB = "aurai"
│
├─ Run
│  └─ uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
│
└─ Models
   ├─ models/best_multilabel.pt
   ├─ models/skin_type_resnet18.pt
   └─ models/per_class_thresholds.json

Frontend (React) Setup:
├─ Node.js 16+
├─ npm install
├─ npm start (dev)
├─ npm run build (production)
└─ API_BASE = "http://127.0.0.1:8000"

Database Setup:
├─ MySQL 5.7+
├─ CREATE DATABASE aurai;
├─ Tables auto-created by init_db()
└─ Credentials in env vars

Production Notes:
├─ ✅ Use bcrypt for password hashing (not SHA256)
├─ ✅ Move secrets to .env file
├─ ✅ Use HTTPS + secure cookies
├─ ✅ Implement rate limiting
├─ ✅ Add logging (not just print)
├─ ✅ Docker containerization recommended
└─ ✅ Validate all user inputs (SQL injection risk)
```

---

## Summary

This AURAI system implements:
1. **Dual-path ML inference**: Skin type classification + multilabel condition detection
2. **Safety-first design**: Quality gates, allergy filtering, personalization
3. **User-centric UI**: React SPA with auth, history tracking
4. **Modular architecture**: Each concern (quality, safety, explainability) is separate
5. **DB persistence**: User profiles + scan history in MySQL

All components follow **clean architecture** principles with clear separation of concerns.
