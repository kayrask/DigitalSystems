# api/recommender.py
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Formulas folder
FORMULAS_DIR = BASE_DIR / "data" / "formulas"

# --- Template mapping (your real files + sheet names) ---
FORMULA_MAP = {
    "normal": {
        "cleanser": ("normal cilt/normal cilt nazik_yuz_temizleyici Rinse off.xlsx", "Formül (INCI)"),
        "toner": ("normal cilt/Normal cilt dengeleyici_tonik_Leave on.xlsx", "Tonik Formül (INCI)"),
        "serum": ("normal cilt/normal cilt hidratif_serum_Leave on.xlsx", "Hidratif Serum (INCI)"),
        "moisturiser": ("normal cilt/normal cilt hafif_nemlendirici_krem.xlsx", "Hafif Nemlendirici Krem (INCI)"),
    },
    "combination": {
        "cleanser": ("karma cilt/karma_cilt_temizleyici.xlsx", "Karma Cilt Temizleyici (CPSR)"),
        "toner": ("karma cilt/karma cilt Gunluk_Denge_Tonik_BHA içermeyen.xlsx", "Formül (INCI)"),
        "serum": ("karma cilt/karma Cilt Jel_Serum_Leave on.xlsx", "Formül (INCI)"),
        "moisturiser": ("karma cilt/karma cilt Ultra_Hafif_Krem_leave on.xlsx", "Formül (INCI)"),
    },
    "oily": {
        "cleanser": ("yağlı cilt/Yagli_Cilt_Serisi_Profesyonel_Uretim_Dosyasi.xlsx", "1_Temizleme_Jeli"),
        "toner": ("yağlı cilt/Yagli_Cilt_Serisi_Profesyonel_Uretim_Dosyasi.xlsx", "2_Tonik"),
        "serum": ("yağlı cilt/Yagli_Cilt_Serisi_Profesyonel_Uretim_Dosyasi.xlsx", "3_Gece_Serumu"),
        "moisturiser": ("yağlı cilt/Yagli_Cilt_Serisi_Profesyonel_Uretim_Dosyasi.xlsx", "4_Nemlendirici"),
    },
}

# --- Rows to ignore (your notes/headers/pH adjusters) ---
NOTE_STARTS = (
    "durum", "gereklilik", "rolü", "hazırlık", "denge", "aktif bakım", "mühürleme"
)
SKIP_CONTAINS = (
    "lactic acid /", "citric acid", "sodium hydroxide",
    "laktik asit", "laktİk asİt", "laktik asit /", "ph"
)

def _is_note_row(s: str) -> bool:
    if not s:
        return True
    t = s.strip().lower()
    if any(t.startswith(x) for x in NOTE_STARTS):
        return True
    if any(x in t for x in SKIP_CONTAINS):
        return True
    # skip long CPSR commentary sentences
    if len(t) > 80 and " " in t:
        return True
    return False


def _to_float(x):
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace(",", ".")
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except:
        return None


def load_formula_excel(path: Path, sheet_name: str) -> List[Dict[str, Any]]:
    """Loads rows: [{'inci': str, 'percent': float or None}]"""
    df = pd.read_excel(path, sheet_name=sheet_name)

    inci_col = None
    pct_col = None

    for c in df.columns:
        cl = str(c).strip().lower()
        if "inci" in cl:
            inci_col = c
        if "yüzde" in cl or "yuzde" in cl or cl == "%" or "%" in cl or "percent" in cl:
            pct_col = c

    if inci_col is None:
        return []

    rows = []
    for _, r in df.iterrows():
        inci = str(r.get(inci_col) or "").strip()
        if not inci or inci.lower() == "nan":
            continue
        if _is_note_row(inci):
            continue

        pct = _to_float(r.get(pct_col)) if pct_col else None
        rows.append({"inci": inci, "percent": pct})

    return rows


# --- Personalization rules based on your templates ---
# We keep conservative max_step so you "change a little"
TWEAK_RULES = {
    "Niacinamide": {
        "max_step": 1.0,
        "driven_by": ["acne", "blackheads", "hyperpigmentation"],
    },
    "Azelaic Acid": {
        "max_step": 1.0,
        "driven_by": ["acne", "blackheads", "hyperpigmentation", "redness"],
    },
    "Panthenol": {
        "max_step": 0.5,
        "driven_by": ["redness", "acne"],
    },
    "Allantoin": {
        "max_step": 0.2,
        "driven_by": ["redness", "acne"],
    },
    "Sodium PCA": {
        "max_step": 0.5,
        "driven_by": ["redness"],
    },
    "Sodium Hyaluronate": {
        "max_step": 0.1,
        "driven_by": ["redness", "bags"],
    },
    "Propanediol": {
        "max_step": 1.0,
        "driven_by": ["hyperpigmentation", "acne"],
    },
    "Glycerin": {
        "max_step": 1.0,
        "driven_by": ["redness"],
    },
}

# Don't change preservative/surfactant/emulsifier system (CPSR stability)
LOCKED_KEYWORDS = [
    "Phenoxyethanol", "Ethylhexylglycerin",
    "Dehydroacetic Acid", "Benzyl Alcohol",
    "Coco-Glucoside", "Decyl Glucoside", "Lauryl Glucoside",
    "Cocamidopropyl Betaine",
    "Cetearyl Olivate", "Sorbitan Olivate",
    "Xanthan Gum",
    "Cetyl Alcohol",
]

WATER_KEYS = {"aqua", "water"}


def _is_locked(inci: str) -> bool:
    t = (inci or "").lower()
    return any(k.lower() in t for k in LOCKED_KEYWORDS)


def personalize_formula(
    formula_rows: List[Dict[str, Any]],
    concern_probs: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (personalized_rows, changes)
    changes: [{'inci':..., 'from':..., 'to':...}]
    """
    updated = [dict(r) for r in formula_rows]
    changes = []

    # pick water row to rebalance
    water_idx = None
    for i, r in enumerate(updated):
        if (r["inci"] or "").strip().lower() in WATER_KEYS:
            water_idx = i
            break
    if water_idx is None:
        # fallback: largest percentage row
        water_idx = max(range(len(updated)), key=lambda i: (updated[i]["percent"] or 0.0))

    # tweak allowed actives
    for r in updated:
        inci = r["inci"]
        pct = r["percent"]
        if pct is None:
            continue
        if _is_locked(inci):
            continue

        rule = TWEAK_RULES.get(inci)
        if not rule:
            continue

        sev = max((concern_probs.get(c, 0.0) for c in rule["driven_by"]), default=0.0)
        if sev < 0.5:
            continue

        base = float(pct)
        scale = (sev - 0.5) / 0.5  # 0..1
        delta = scale * rule["max_step"]
        new_pct = round(base + delta, 3)

        if new_pct != base:
            r["percent"] = new_pct
            changes.append({"inci": inci, "from": base, "to": new_pct})

    # rebalance water to 100
    total = sum((x["percent"] or 0.0) for x in updated)
    diff = 100.0 - total
    water_before = updated[water_idx]["percent"] or 0.0
    updated[water_idx]["percent"] = round(water_before + diff, 3)

    if round(updated[water_idx]["percent"], 3) != round(water_before, 3):
        changes.append({
            "inci": updated[water_idx]["inci"],
            "from": round(water_before, 3),
            "to": round(updated[water_idx]["percent"], 3)
        })

    return updated, changes


def recommend_routine(model_results: Dict[str, Dict[str, Any]], skin_type: str = "combination") -> Dict[str, Any]:
    """
    Template-locked routine + small % tweaks based on concern probabilities.
    """
    concern_probs = {k: float(v.get("probability", 0.0)) for k, v in model_results.items()}

    st = (skin_type or "combination").lower()
    if st not in FORMULA_MAP:
        st = "combination"

    steps = ["cleanser", "toner", "serum", "moisturiser"]
    out_steps = []

    for step in steps:
        rel_path, sheet = FORMULA_MAP[st][step]
        file_path = FORMULAS_DIR / rel_path

        base_formula = load_formula_excel(file_path, sheet_name=sheet)
        personalized, changes = personalize_formula(base_formula, concern_probs)

        out_steps.append({
            "step": step,
            "template_file": str((BASE_DIR / "data" / "formulas" / rel_path).relative_to(BASE_DIR)),
            "sheet": sheet,
            "focus_concerns": sorted(concern_probs, key=concern_probs.get, reverse=True)[:3],
            "formula_base": base_formula,
            "formula_personalized": personalized,
            "changes": changes,
        })

    return {
        "skin_type": st,
        "steps": out_steps,
    }
