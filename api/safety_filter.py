# api/safety_filter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Keep aligned with your recommender "locked system" idea
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

# Common allergy aliases -> search tokens
ALLERGY_ALIASES = {
    "fragrance": ["parfum", "fragrance", "aroma"],
    "niacinamide": ["niacinamide"],
    "nuts": ["sweet almond", "prunus amygdalus", "argania spinosa", "macadamia", "hazelnut"],
    "lanolin": ["lanolin"],
}

@dataclass
class SafetyConfig:
    # If true, allow removing non-locked allergens by setting % to 0 and rebalancing water
    allow_remove_nonlocked: bool = True


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _tokenize_allergies(allergies_text: str | None) -> List[str]:
    """
    Turns user allergies into search tokens.
    "fragrance, nuts" -> ["parfum","fragrance","aroma",...]
    """
    if not allergies_text:
        return []

    raw = [x.strip().lower() for x in allergies_text.replace("\n", ",").split(",") if x.strip()]
    tokens: List[str] = []
    for item in raw:
        if item in ALLERGY_ALIASES:
            tokens += ALLERGY_ALIASES[item]
        else:
            # also keep literal user token
            tokens.append(item)
    # unique, stable order
    seen = set()
    out = []
    for t in tokens:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _is_locked_inci(inci: str) -> bool:
    t = _norm(inci)
    return any(_norm(k) in t for k in LOCKED_KEYWORDS)


def _find_water_index(rows: List[Dict[str, Any]]) -> int:
    for i, r in enumerate(rows):
        if _norm(r.get("inci")) in WATER_KEYS:
            return i
    # fallback to largest percent row
    best_i = 0
    best_val = -1.0
    for i, r in enumerate(rows):
        pct = r.get("percent")
        try:
            v = float(pct) if pct is not None else 0.0
        except Exception:
            v = 0.0
        if v > best_val:
            best_val = v
            best_i = i
    return best_i


def _rebalance_to_100(rows: List[Dict[str, Any]], water_idx: int) -> None:
    total = 0.0
    for r in rows:
        pct = r.get("percent")
        total += float(pct) if pct is not None else 0.0
    diff = 100.0 - total
    before = float(rows[water_idx].get("percent") or 0.0)
    rows[water_idx]["percent"] = round(before + diff, 3)


def filter_routine_for_user(
    routine: Dict[str, Any],
    allergies_text: str | None,
    cfg: SafetyConfig | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (filtered_routine, safety_report)
    safety_report includes per-step actions + warnings.
    """
    cfg = cfg or SafetyConfig()
    tokens = _tokenize_allergies(allergies_text)

    safety_report: Dict[str, Any] = {
        "allergy_tokens": tokens,
        "steps": [],
        "warnings": [],
    }

    if not tokens or not routine or not routine.get("steps"):
        return routine, safety_report

    new_steps = []
    for step in routine["steps"]:
        step_out = dict(step)
        step_report = {
            "step": step.get("step"),
            "status": "ok",
            "removed": [],
            "blocked_matches": [],
            "notes": [],
        }

        rows = step.get("formula_personalized") or []
        if not rows:
            safety_report["steps"].append(step_report)
            new_steps.append(step_out)
            continue

        updated = [dict(r) for r in rows]
        water_idx = _find_water_index(updated)

        # Scan for allergen matches
        for r in updated:
            inci = r.get("inci") or ""
            inci_l = _norm(inci)

            matched = [t for t in tokens if t in inci_l]
            if not matched:
                continue

            if _is_locked_inci(inci):
                # can't safely remove; block the step
                step_report["status"] = "blocked"
                step_report["blocked_matches"].append({"inci": inci, "matched": matched})
            else:
                if cfg.allow_remove_nonlocked:
                    before = float(r.get("percent") or 0.0)
                    if before > 0:
                        r["percent"] = 0.0
                        step_report["removed"].append({"inci": inci, "matched": matched, "from": before, "to": 0.0})
                else:
                    step_report["status"] = "warn"
                    step_report["notes"].append(f"Allergen present: {inci}")

        # If blocked, keep formula unchanged but flag it (safer academically)
        if step_report["status"] == "blocked":
            safety_report["warnings"].append(
                f"{step_report['step']}: contains locked-system ingredient matching user allergy; step blocked."
            )
            step_out["safety_status"] = "blocked"
            step_out["safety_report"] = step_report
            new_steps.append(step_out)
            safety_report["steps"].append(step_report)
            continue

        # If we removed anything, rebalance to 100
        if step_report["removed"]:
            _rebalance_to_100(updated, water_idx)

            # append to existing changes so History shows it nicely
            existing_changes = list(step_out.get("changes") or [])
            for rm in step_report["removed"]:
                existing_changes.append({"inci": rm["inci"], "from": rm["from"], "to": rm["to"]})
            step_out["changes"] = existing_changes
            step_out["formula_personalized"] = updated
            step_out["safety_status"] = "adjusted"
        else:
            step_out["safety_status"] = "ok"

        step_out["safety_report"] = step_report
        new_steps.append(step_out)
        safety_report["steps"].append(step_report)

    out_routine = dict(routine)
    out_routine["steps"] = new_steps
    return out_routine, safety_report
