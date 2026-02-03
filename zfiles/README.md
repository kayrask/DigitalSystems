# AURAI – Training, Evaluation, and Running the System

## 1) Prerequisites
- Python 3.11
- Node 18+ (for the React frontend)
- MySQL running locally (defaults: host `localhost`, port `3306`, user `root`, password `kayra0505`, db `aurai`). Update these in `api/main.py` if needed.

## 2) Python environment & dependencies
```bash
cd "/Users/kayra/Developer/DISSERTATION PROJECT"
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
pip install -r api/requirements.txt   
```

## 3) Data locations (default)
- Multilabel metadata CSV: `data/processed/all_multilabel.csv` (with columns: `image_path` or `image`, `split`, and one-hot cols for acne/bags/blackheads/hyperpigmentation/redness)
- Multilabel images root: project root (`.`) unless your CSV paths are absolute; adjust with `--images_root`.
- Skin-type dataset root: `data/skin_type_dataset/` with `train/`, `validate/` (or `val/`), and `test/` subfolders.

## 4) Train – Skin Concerns (multilabel)
```bash
source .venv/bin/activate
python scripts/train_multilabel.py \
  --metadata data/processed/all_multilabel.csv \
  --images_root . \
  --epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --out_dir models
```
- Outputs: checkpoint in `models/` (e.g., `best_multilabel.pt`) and any logs produced by the script.
- If you need backbone freeze: add `--freeze_backbone`.

## 5) Train – Skin Type (5-class)
```bash
source .venv/bin/activate
python scripts/train_skin_type.py \
  --data_dir data/skin_type_dataset \
  --epochs 15 \
  --batch_size 32 \
  --lr 3e-4 \
  --out models/skin_type_resnet18.pt
```
- Ensure `train/`, `validate/` (or `val/`), and `test/` exist under `data/skin_type_dataset/`.

## 6) Evaluate – Skin Concerns (multilabel)
```bash
source .venv/bin/activate
python scripts/eval_skin.py \
  --metadata data/processed/all_multilabel.csv \
  --images_root . \
  --checkpoint models/best_multilabel.pt \
  --threshold 0.5
```
- Prints classification report, ROC-AUC/AP, and per-class metrics.
- Update `--threshold` if you have per-class thresholds (see `models/per_class_thresholds.json`).

## 7) Evaluate – Skin Type (5-class)
Edit `scripts/eval_skin_type.py` constants or export env vars, then run:
```bash
source .venv/bin/activate
python scripts/eval_skin_type.py
```
Defaults inside the script:
- `MODEL_PATH = "models/skin_type_resnet18.pt"`
- `DATA_DIR = "data/skin_type_dataset/test"`
- `BATCH_SIZE = 32`, `IMG_SIZE = 224`

## 8) Run the FastAPI backend
```bash
source .venv/bin/activate
# Ensure MySQL is up and the DB `aurai` exists (adjust credentials in api/main.py if needed)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
- Startup will create tables if missing (users, scans).
- Main predict endpoint: `POST /predict` with form-data `file` (image). Other routes: `/login`, `/register`, `/history`, `/profile`.

## 9) Run the React frontend
```bash
cd frontend
npm install
npm start
```
- The app expects the backend at `http://127.0.0.1:8000`. If you change ports/hosts, update API_BASE in the frontend pages/components that call the API.

## 10) Model artifacts
- Trained weights live in `models/` (e.g., `best_multilabel.pt`, `best_multilabel_for_eval.pt`, `per_class_thresholds.json`, `skin_type_resnet18.pt`).
- Ensure the backend points to the correct model paths; adjust in `api/inference.py` or wherever you load them.

## 11) Quick smoke test
- Backend: `curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/face.jpg"`
- Frontend: open `http://localhost:3000`, log in/register, go to “Get a new scan”, upload an image.

## 12) Troubleshooting
- File watcher errors about missing folders: ensure `files/` exists (some tools watch it). `mkdir -p files` fixes it.
- MySQL connection refused: start MySQL and confirm credentials/DB name in `api/main.py`.
- MPS/CUDA vs CPU: scripts auto-select MPS (Apple), else CUDA, else CPU.
