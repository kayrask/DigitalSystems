# AURAI - Detailed Data Flow & System Specifications

## 1. DATA FLOW THROUGH PREDICTION PIPELINE

### Input Stage
```
User Image (PNG/JPEG)
    ↓
[PIL Image.open & convert to RGB]
    ↓
RAW: PIL.Image (H, W, 3) RGB
    ├─ Width: variable
    ├─ Height: variable
    └─ Mode: RGB (uint8 0-255)
```

### Quality Gate Stage
```
INPUT: PIL.Image
OUTPUT: {
  "passed": bool,
  "reasons": List[str],  // e.g., ["too_dark", "low_contrast"]
  "metrics": {
    "width": int,
    "height": int,
    "short_side": int,
    "brightness": float (0..1),  // 0.299*R + 0.587*G + 0.114*B
    "contrast": float,           // std(grayscale)
    "blur_var": float            // variance(Laplacian response)
  },
  "thresholds": {...}
}

Failure Reasons:
├─ "low_resolution" - min_short_side < 320px
├─ "too_dark" - brightness < 0.20
├─ "too_bright" - brightness > 0.85
├─ "low_contrast" - contrast < 0.05
└─ "blurry" - blur_var < 0.0015
```

### Face Detection & Cropping Stage
```
INPUT: PIL.Image (full face photo)
    ↓
[MediaPipe FaceDetection.process()]
    ↓
OUTPUT: {
  "face_found": bool,
  "score": float (0..1),  // detection confidence
  "bbox": {
    "x": int,  // pixel offset in original image
    "y": int,
    "w": int,  // width of expanded bounding box
    "h": int   // height of expanded bounding box
  }
}

Processing:
├─ Detect face with MediaPipe (model_selection=0, min_confidence=0.6)
├─ Find best detection by score
├─ Extract relative bbox (xmin, ymin, width, height)
├─ Expand by margin=20% in all directions
├─ Clamp to image boundaries
└─ Return cropped PIL.Image
```

### Oval Mask Stage
```
INPUT: PIL.Image (cropped face)
    ↓
[Create elliptical mask + Gaussian blur]
    ↓
OUTPUT: PIL.Image (masked + feathered)

Processing:
├─ Create binary mask (0 outside, 255 inside oval)
├─ Apply Gaussian blur (σ=18) to feather edges
├─ Create background: Gaussian blur(σ=10) of original
├─ Composite: keep oval from original, use blurred outside
└─ Soften transitions for natural appearance
```

### Skin Type Model Stage
```
INPUT: PIL.Image (cropped + masked, any size)
    ↓
[Resize to 224×224]
    ↓
[ToTensor() → normalize]
    ↓
tensor: [1, 3, 224, 224] float32
    ├─ min: 0.0 (after normalization)
    ├─ max: ~2.64 (3σ above normalized mean)
    └─ mean: ~0.0 (normalized by ImageNet stats)
    ↓
[ResNet18 forward pass]
    ├─ Input: [1, 3, 224, 224]
    ├─ Output logits: [1, 5]  // 5 classes
    └─ Processing: conv→maxpool→residual blocks→avgpool→fc
    ↓
[Softmax across 5 classes]
    ↓
OUTPUT: {
  "skin_type": str,  // "normal", "oily", "dry", "combination", "sensitive"
  "confidence": float (0..1),
  "probs": {
    "normal": 0.XX,
    "oily": 0.XX,
    "dry": 0.XX,
    "combination": 0.XX,
    "sensitive": 0.XX
    // sum = 1.0
  }
}
```

### Skin Concerns Model Stage
```
INPUT: PIL.Image (same as skin type)
    ↓
[Same preprocessing: resize, tensor, normalize]
    ↓
tensor: [1, 3, 224, 224]
    ↓
[ResNet18 forward pass]
    ├─ Output logits: [1, 5]  // 5 independent labels
    └─ Labels: [acne, bags, blackheads, hyperpigmentation, redness]
    ↓
[Apply thresholds per-class]
    ↓
OUTPUT per-label:
{
  "acne": {
    "probability": 0.82,  // sigmoid(logit) ∈ [0, 1]
    "prediction": 1       // 1 if prob >= threshold[acne], else 0
  },
  "bags": {
    "probability": 0.12,
    "prediction": 0
  },
  ...
}

Thresholds from per_class_thresholds.json:
{
  "acne": 0.45,
  "bags": 0.35,
  "blackheads": 0.40,
  "hyperpigmentation": 0.50,
  "redness": 0.38
}
```

### Recommendation Engine Stage
```
INPUT:
{
  "concern_probs": {
    "acne": 0.82,
    "bags": 0.12,
    "blackheads": 0.45,
    "hyperpigmentation": 0.33,
    "redness": 0.19
  },
  "skin_type": "combination"
}

[Extract top concerns (prob ≥ 0.5)]
├─ Top 1: acne (0.82)
├─ Top 2: blackheads (0.45) — below 0.5, included anyway
└─ Top 3: (none above next threshold)

[Load ingredient database from Excel]
├─ Source: data/knowledge/cilt_tipleri.xlsx
├─ Sheet: "hammaddeler"
├─ Columns: name, function_tr, notes
└─ Parse → ING_DB = {
     "niacinamide": {
       "name": "Niacinamide",
       "function_tr": "sebum düzenleyici, anti-enflam",
       "notes": "..."
     },
     ...
   }

[Score ingredients by concern relevance]
For each ingredient:
  score = 0
  for concern in top_concerns:
    if concern in CONCERN_KEYWORDS:
      keywords = CONCERN_KEYWORDS[concern]
      if any(kw in ingredient_text):
        score += concern_prob[concern]

Example:
  Niacinamide:
    concern: acne (0.82) → "sebum düzenleyici" matches "sebum" → +0.82
    concern: blackheads (0.45) → no match
    score = 0.82
    
  Salicylic Acid:
    concern: acne (0.82) → "antibakteriyel" matches ✓ → +0.82
    concern: blackheads (0.45) → "gözenek, komedon" matches ✓ → +0.45
    score = 1.27

[Return top-K ingredients by score]
OUTPUT:
{
  "skin_type": "combination",
  "top_concerns": ["acne", "blackheads"],
  "suggested_ingredients": [
    "niacinamide",
    "salicylic acid",
    "panthenol",
    ...  // top 10
  ]
}
```

### Formula Personalization Stage
```
INPUT:
{
  "formula_rows": [
    {"inci": "aqua", "percent": 78.0},
    {"inci": "niacinamide", "percent": 2.0},
    {"inci": "salicylic acid", "percent": 1.0},
    {"inci": "glycerin", "percent": 5.0},
    {"inci": "xanthan gum", "percent": 0.5},
    ...
  ],
  "concern_probs": {"acne": 0.82, "blackheads": 0.45, ...},
  "skin_type": "combination"
}

[For each TWEAKABLE active:]
TWEAKABLE = {
  "niacinamide": {
    "concerns": ["acne", "blackheads", "hyperpigmentation"],
    "max_step": 1.0  // max increase from base
  },
  "salicylic acid": {
    "concerns": ["acne", "blackheads"],
    "max_step": 0.5
  },
  ...
}

[Find severity for ingredient:]
For "niacinamide":
  severity = max([
    concern_probs.get("acne", 0) = 0.82,
    concern_probs.get("blackheads", 0) = 0.45,
    concern_probs.get("hyperpigmentation", 0) = 0.33
  ]) = 0.82

[If severity ≥ 0.5, compute new concentration:]
  base = 2.0
  max_step = 1.0
  
  // Map severity 0.5→1.0 to increase 0→max_step
  target = base + (severity - 0.5) / 0.5 * max_step
         = 2.0 + (0.82 - 0.5) / 0.5 * 1.0
         = 2.0 + 0.64
         = 2.64%

[Check usage bounds from Excel:]
  min_usage, max_usage = parse_range("0.5-3")
  target = clamp(target, min_usage, max_usage)
         = clamp(2.64, 0.5, 3.0)
         = 2.64% ✓

[Find water row and rebalance:]
  water_idx = find_water_index(formula_rows)  // usually index 0
  
  new_rows = [
    {"inci": "aqua", "percent": 76.36},  // 78 - (2.64-2.0)
    {"inci": "niacinamide", "percent": 2.64},
    ...
  ]
  // Ensure sum = 100%

OUTPUT: personalized formula rows with updated percents
```

### Safety Filter Stage
```
INPUT:
{
  "routine": {...},
  "user_allergies_text": "fragrance, niacinamide"
}

[Tokenize allergies:]
ALLERGY_ALIASES = {
  "fragrance": ["parfum", "fragrance", "aroma"],
  "niacinamide": ["niacinamide"],
  ...
}

tokens = ["parfum", "fragrance", "aroma", "niacinamide"]

[For each ingredient in routine:]
LOCKED_KEYWORDS = [
  "Phenoxyethanol", "Ethylhexylglycerin", "Dehydroacetic Acid",
  "Benzyl Alcohol", "Coco-Glucoside", ...
]

For "Niacinamide" (2.64%):
  ├─ Is locked? No (not in LOCKED_KEYWORDS)
  ├─ User allergic? Yes ("niacinamide" in tokens)
  ├─ Allow removal? Yes (allow_remove_nonlocked=True)
  └─ Action: Set percent = 0 (mark for removal)

For "Phenoxyethanol" (0.8%):
  ├─ Is locked? Yes (preservative)
  ├─ User allergic? No
  └─ Action: Keep as-is (locked, cannot remove)

For "Fragrance" (0.3%):
  ├─ Is locked? No
  ├─ User allergic? Yes ("parfum" matches "fragrance")
  └─ Action: Set percent = 0

[Rebalance water to 100%:]
  Original sum: 100%
  After removing items: 99.24%
  Diff: +0.76%
  Water new: 78 + 0.76 = 78.76%

OUTPUT: cleaned routine with allergen-free formula
```

### Outcome & Risk Analysis Stage
```
INPUT:
{
  "routine": {...personalized & filtered...},
  "concern_probs": {"acne": 0.82, ...},
  "skin_type": "combination"
}

[Extract all INCI names from routine:]
inci_tokens = ["aqua", "niacinamide", "panthenol", ...]

[Score benefit based on concern coverage:]
ACTIVE_BENEFIT = {
  "acne": [
    ("salicylic", 0.35),
    ("bha", 0.25),
    ("niacinamide", 0.20),
    ...
  ],
  ...
}

benefit_score = 0
for token in inci_tokens:
  for concern, severity in concern_probs.items():
    for active, weight in ACTIVE_BENEFIT[concern]:
      if active in token:
        contribution = weight * severity
        benefit_score += contribution

Example:
  Token "niacinamide" found:
    ├─ acne (0.82) + niacinamide (0.20) → +0.164
    ├─ blackheads (0.45) + niacinamide (0.10) → +0.045
    └─ hyperpigmentation (0.33) + niacinamide (0.18) → +0.059
  subtotal = 0.268

[Score risk based on irritants & skin type:]
IRRITANT_FLAGS = [
  ("fragrance", 0.18),
  ("alcohol denat", 0.14),
  ("retinol", 0.22),
  ...
]

STRONG_ACTIVES = [("retinol", 0.22), ("salicylic", 0.18), ...]

base_risk = Phase3Config.base_risk = 0.10

For skin_type "combination":
  multiplier = 1.0 (baseline)

risk_score = base_risk
for token in inci_tokens:
  for irritant, weight in IRRITANT_FLAGS:
    if irritant in token:
      risk_score += weight * multiplier

final_risk = clamp(risk_score, 0, 1.0)

OUTPUT:
{
  "benefit_score": 0.XX (0..1),
  "risk_score": 0.XX (0..1),
  "recommendation": "high_benefit_low_risk" | "proceed_with_caution" | ...
}
```

### Explainability (GradCAM) Stage
```
INPUT:
  model: SkinModel
  image: PIL.Image
  label_idx: int (e.g., 0 for acne)

[Run forward pass with hooks:]
  ├─ Capture last conv layer activations: [1, 512, 7, 7]
  ├─ Capture model output logits: [1, 5]
  └─ Compute gradients w.r.t. conv activations

[Compute Class Activation Map:]
  cam = sum(weights[c] * activations) for c in top K
      # weights = dlogits[label_idx] / d(conv_out)
      # Shape: [7, 7]

[Normalize & enhance visibility:]
  ├─ Clip to [0, 1]
  ├─ Percentile rescale (70th to 99th percentile)
  ├─ Apply gamma correction (^0.6)
  ├─ Apply elliptical soft mask (face-centered)
  └─ Rescale alpha channel

[Create RGBA overlay:]
  rgba = zeros([H, W, 4], dtype=uint8)
  rgba[..., 0] = 255  # RED channel
  rgba[..., 3] = alpha  # alpha from CAM

[Encode to base64 PNG:]
  img.save(buffer, format='PNG')
  base64_str = base64.b64encode(buffer.getvalue()).decode()
  # e.g., "iVBORw0KGgoAAAANS..."

OUTPUT: base64 PNG string for frontend display
```

---

## 2. DATABASE SCHEMA IN DETAIL

### Users Table
```sql
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(64) NOT NULL,  -- SHA256 hex (64 chars)
  phone VARCHAR(50) NULL,
  age INT NULL,
  address TEXT NULL,
  allergies TEXT NULL,  -- e.g., "fragrance, niacinamide, nuts"
  
  INDEXES: (email)  -- for quick login lookups
);
```

### Scans Table
```sql
CREATE TABLE scans (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  results_json TEXT NOT NULL,
  
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEXES: (user_id, created_at DESC)  -- for scan history queries
);

-- results_json structure (typical)
{
  "results": {
    "acne": {"probability": 0.82, "prediction": 1},
    "bags": {"probability": 0.12, "prediction": 0},
    "blackheads": {"probability": 0.45, "prediction": 1},
    "hyperpigmentation": {"probability": 0.33, "prediction": 0},
    "redness": {"probability": 0.19, "prediction": 0}
  },
  "skin_type": {
    "skin_type": "combination",
    "confidence": 0.94,
    "probs": {
      "combination": 0.94,
      "normal": 0.03,
      "oily": 0.02,
      "dry": 0.01,
      "sensitive": 0.00
    }
  },
  "routine": {
    "skin_type": "combination",
    "top_concerns": ["acne", "blackheads"],
    "suggested_ingredients": ["niacinamide", "salicylic acid", ...],
    "steps": [
      {
        "name": "Cleanser",
        "formula_personalized": [
          {"inci": "aqua", "percent": 78.0},
          {"inci": "niacinamide", "percent": 2.64},
          ...
        ]
      }
    ]
  },
  "quality": {
    "passed": true,
    "metrics": {...}
  },
  "risk_analysis": {
    "benefit_score": 0.67,
    "risk_score": 0.15,
    "recommendation": "high_benefit_low_risk"
  },
  "explainability": {
    "heatmap_base64": "iVBORw0KGgo..."
  }
}
```

---

## 3. MODEL SPECIFICATIONS

### SkinModel (Multilabel)
```
Architecture: ResNet18 (pretrained ImageNet) + custom FC
├─ Input: [B, 3, 224, 224]
├─ Backbone: ResNet18 (conv layers → avgpool → fc_input)
│   └─ conv1(64) → layer1 → layer2 → layer3 → layer4
│   └─ avgpool: [B, 512, 1, 1]
├─ FC head: Linear(512, 5)
│   └─ Output: [B, 5] logits
└─ Activation: Sigmoid (independent per-class predictions)

Loss: BCEWithLogitsLoss(logits, y_multilabel)

Thresholds: per_class_thresholds.json
{
  "acne": 0.45,
  "bags": 0.35,
  "blackheads": 0.40,
  "hyperpigmentation": 0.50,
  "redness": 0.38
}

Inference:
  probs = sigmoid(logits)  # [B, 5] in [0, 1]
  predictions = (probs >= thresholds).int()  # [B, 5] binary
```

### SkinTypeModel (Multiclass)
```
Architecture: ResNet18 (pretrained) + custom FC
├─ Input: [B, 3, 224, 224]
├─ Backbone: same ResNet18
├─ FC head: Linear(512, 5)
│   └─ Output: [B, 5] logits
└─ Activation: Softmax(dim=1)

Loss: CrossEntropyLoss(logits, y_class)

Classes (mutually exclusive):
  ["combination", "dry", "normal", "oily", "sensitive"]

Inference:
  probs = softmax(logits)  # [B, 5], sum to 1.0
  prediction = argmax(probs)  # index of max prob
```

---

## 4. API REQUEST/RESPONSE EXAMPLES

### POST /predict
```
REQUEST (multipart/form-data):
  file: <binary image data>

RESPONSE (200 OK):
{
  "results": {
    "acne": {"probability": 0.82, "prediction": 1},
    ...
  },
  "skin_type": {
    "skin_type": "combination",
    "confidence": 0.94,
    "probs": {...}
  },
  "routine": {...},
  "quality": {...},
  "risk_analysis": {...},
  "explainability": {...}
}
```

### POST /register
```
REQUEST:
{
  "name": "Alice Smith",
  "email": "alice@example.com",
  "password": "SecurePassword123"
}

RESPONSE (200 OK):
{
  "id": 42,
  "name": "Alice Smith",
  "email": "alice@example.com"
}

ERROR (400 Conflict):
{
  "detail": "Email already registered"
}
```

### POST /login
```
REQUEST:
{
  "email": "alice@example.com",
  "password": "SecurePassword123"
}

RESPONSE (200 OK):
{
  "id": 42,
  "name": "Alice Smith",
  "email": "alice@example.com"
}

ERROR (400 Unauthorized):
{
  "detail": "Invalid email or password"
}
```

### GET /scans?user_id=42&limit=10
```
RESPONSE (200 OK):
{
  "items": [
    {
      "id": 101,
      "created_at": "2025-01-15T14:32:00",
      "results": {...full results_json...}
    },
    {
      "id": 100,
      "created_at": "2025-01-14T10:15:00",
      "results": {...}
    }
  ]
}
```

---

## 5. TRAINING PIPELINE

### Multilabel Training
```
1. Load data:
   CSV: data/processed/all_multilabel.csv
   Columns: [image_path, split, acne, bags, blackheads, hyperpigmentation, redness]
   Example row:
     data/raw_a/acne/0/img_001.jpg, train, 1, 0, 1, 0, 0

2. Augmentation (train only):
   ├─ RandomResizedCrop(224, scale=[0.78, 1.0])
   ├─ RandomHorizontalFlip(0.5)
   ├─ ColorJitter(brightness=0.22, contrast=0.22, ...)
   ├─ RandomPerspective(distortion_scale=0.12)
   ├─ GaussianBlur(kernel_size=3, σ=[0.1, 1.2])
   ├─ RandomErasing(p=0.35)
   └─ Normalize(ImageNet stats)

3. Training loop:
   for epoch in range(30):
     for batch in train_loader:
       images, labels = batch
       logits = model(images)  # [B, 5]
       loss = BCEWithLogitsLoss(logits, labels.float())
       loss.backward()
       optimizer.step()
     
     # Validation
     for batch in val_loader:
       logits = model(batch[0])
       probs = sigmoid(logits)
       f1 = micro_f1(probs, batch[1])
       save if best

4. Output:
   models/best_multilabel.pt
   {
     "model": model.state_dict(),
     ...
   }
```

### Skin Type Training
```
1. Load data:
   ImageFolder: data/skin_type_dataset/train/
   ├─ Combination/
   ├─ Dry/
   ├─ Normal/
   ├─ Oily/
   └─ Sensitive/

2. Augmentation (train only):
   ├─ Resize(224)
   ├─ RandomHorizontalFlip(0.5)
   ├─ ColorJitter(0.1, 0.1, 0.1, 0.02)
   └─ Normalize

3. Training:
   for epoch in range(15):
     for batch in train_loader:
       images, labels = batch  # labels: class indices [0-4]
       logits = model(images)  # [B, 5]
       loss = CrossEntropyLoss(logits, labels)
       loss.backward()
       optimizer.step()
     
     val_acc = evaluate(val_loader)
     if best: save checkpoint

4. Output:
   models/skin_type_resnet18.pt
   {
     "model": model.state_dict(),
     "classes": ["combination", "dry", "normal", "oily", "sensitive"]
   }
```

### Threshold Finding
```
1. Load best_multilabel checkpoint
2. Run on validation/test set:
   for img, y_true in test_set:
     logits = model(img)
     probs = sigmoid(logits)  # per-label probs
     # Store for analysis

3. For each label, find optimal threshold:
   For acne:
     optimal_threshold = max_f1_score(y_true_acne, probs_acne)
     # e.g., 0.45

4. Save thresholds:
   models/per_class_thresholds.json
   {
     "acne": 0.45,
     "bags": 0.35,
     ...
   }
```

---

## 6. CONSTANTS & CONFIGURATION

### Concern Keywords (Recommender)
```python
CONCERN_KEYWORDS = {
  "acne": ["sebum", "sivilce", "akne", "antibakteriyel", "sebum düzenleyici"],
  "blackheads": ["gözenek", "komedon", "sebum"],
  "hyperpigmentation": ["aydınlat", "leke", "pigment"],
  "redness": ["yatıştır", "kızarıklık", "hassas", "anti enflam"],
  "bags": ["göz", "ödem", "kafein", "mikrosirkülasyon"],
}
```

### Tweakable Ingredients
```python
TWEAKABLE = {
  "niacinamide": {
    "concerns": ["acne", "blackheads", "hyperpigmentation"],
    "max_step": 1.0
  },
  "alpha arbutin": {
    "concerns": ["hyperpigmentation"],
    "max_step": 0.5
  },
  ...
}
```

### Active Benefit Scoring
```python
ACTIVE_BENEFIT = {
  "acne": [
    ("salicylic", 0.35),
    ("bha", 0.25),
    ("niacinamide", 0.20),
    ...
  ],
  ...
}
```

### Safety Config
```python
LOCKED_KEYWORDS = [
  "Phenoxyethanol", "Ethylhexylglycerin",
  "Dehydroacetic Acid", "Benzyl Alcohol",
  "Xanthan Gum", "Cetyl Alcohol",
  ...
]

ALLERGY_ALIASES = {
  "fragrance": ["parfum", "fragrance", "aroma"],
  "nuts": ["sweet almond", "macadamia", "hazelnut"],
  ...
}
```

This detailed documentation provides complete traceability through all data transformations in the AURAI system.
