# api/outcome_risk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# --- Simple keyword maps (you can expand over time) ---

ACTIVE_BENEFIT = {
    # concern -> list of (token, weight)
    "acne": [
        ("salicylic", 0.35), ("bha", 0.25), ("lactic acid", 0.15), ("glycolic", 0.15),
        ("benzoyl", 0.35), ("niacinamide", 0.20), ("zinc pca", 0.15),
        ("tea tree", 0.08),
    ],
    "blackheads": [
        ("salicylic", 0.40), ("bha", 0.25), ("glycolic", 0.15), ("lactic acid", 0.10),
        ("niacinamide", 0.10),
    ],
    "redness": [
        ("panthenol", 0.18), ("allantoin", 0.18), ("centella", 0.18),
        ("madecassoside", 0.18), ("niacinamide", 0.14), ("beta-glucan", 0.16),
        ("bisabolol", 0.12),
    ],
    "hyperpigmentation": [
        ("niacinamide", 0.18), ("alpha arbutin", 0.25), ("arbutin", 0.20),
        ("tranexamic", 0.30), ("azelaic", 0.25), ("vitamin c", 0.22),
        ("ascorb", 0.22), ("kojic", 0.25), ("licorice", 0.15),
        ("glycolic", 0.18),
    ],
    "bags": [
        ("caffeine", 0.25), ("peptide", 0.15), ("niacinamide", 0.10),
        ("hyaluron", 0.10),
    ],
}

IRRITANT_FLAGS = [
    ("parfum", 0.18), ("fragrance", 0.18), ("aroma", 0.18),
    ("essential oil", 0.12),
    ("alcohol denat", 0.14),
    ("menthol", 0.18), ("eucalypt", 0.10),
]

# actives that can increase irritation risk (especially for sensitive/dry)
STRONG_ACTIVES = [
    ("retinol", 0.22), ("retinal", 0.25), ("tretin", 0.30),
    ("salicylic", 0.18), ("bha", 0.18),
    ("glycolic", 0.20), ("lactic acid", 0.16), ("aha", 0.18),
    ("azelaic", 0.10), ("vitamin c", 0.10), ("ascorb", 0.10),
]

@dataclass
class Phase3Config:
    # How much severity influences benefit (0..1)
    severity_weight: float = 0.55
    # How much ingredient coverage influences benefit
    coverage_weight: float = 0.45

    # risk weights
    base_risk: float = 0.10  # baseline “any skincare can irritate”
    sensitive_multiplier: float = 1.30
    dry_multiplier: float = 1.15


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _extract_inci_tokens(routine: Dict[str, Any]) -> List[str]:
    """
    Collect INCI names from routine steps.
    Expects step["formula_personalized"] as list of {inci, percent}.
    """
    tokens: List[str] = []
    for step in (routine or {}).get("steps", []) or []:
        for row in (step.get("formula_personalized") or []):
            inci = row.get("inci") or ""
            if inci:
                tokens.append(_norm(inci))
    return tokens


def _has_token(tokens: List[str], needle: str) -> bool:
    n = _norm(needle)
    return any(n in t for t in tokens)


def _benefit_score_for_concern(
    concern: str,
    severity_prob: float,
    inci_tokens: List[str],
    cfg: Phase3Config
) -> Tuple[int, List[str]]:
    """
    Returns (0..100 score, drivers list)
    """
    drivers: List[str] = []

    # severity component
    sev = max(0.0, min(float(severity_prob or 0.0), 1.0))  # 0..1

    # ingredient coverage component
    coverage = 0.0
    for token, w in ACTIVE_BENEFIT.get(concern, []):
        if _has_token(inci_tokens, token):
            coverage += w
            drivers.append(f"contains {token}")

    # normalize coverage to 0..1-ish
    coverage = max(0.0, min(coverage, 1.0))

    score = (cfg.severity_weight * sev + cfg.coverage_weight * coverage) * 100.0
    return int(round(max(0.0, min(score, 100.0)))), drivers[:6]


def _risk_score(
    inci_tokens: List[str],
    skin_type: str | None,
    cfg: Phase3Config
) -> Tuple[int, str, List[str]]:
    drivers: List[str] = []
    risk = cfg.base_risk

    # flags
    for token, w in IRRITANT_FLAGS:
        if _has_token(inci_tokens, token):
            risk += w
            drivers.append(f"irritant flag: {token}")

    # strong actives
    strong_hits = 0
    for token, w in STRONG_ACTIVES:
        if _has_token(inci_tokens, token):
            risk += w
            strong_hits += 1
            drivers.append(f"strong active: {token}")

    # “active load” bump
    if strong_hits >= 2:
        risk += 0.08
        drivers.append("multiple strong actives")

    st = _norm(skin_type or "")
    if st == "sensitive":
        risk *= cfg.sensitive_multiplier
        drivers.append("skin type: sensitive")
    elif st == "dry":
        risk *= cfg.dry_multiplier
        drivers.append("skin type: dry")

    # clamp 0..1
    risk = max(0.0, min(risk, 1.0))
    score = int(round(risk * 100))

    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return score, level, drivers[:8]


def compute_outcome_and_risk(
    results: Dict[str, Any],
    skin_type_out: Dict[str, Any] | None,
    routine_filtered: Dict[str, Any],
    cfg: Phase3Config | None = None,
) -> Dict[str, Any]:
    """
    results: your model output dict like {acne: {probability, prediction}, ...}
    skin_type_out: {skin_type, confidence, ...}
    routine_filtered: routine after Phase 2
    """
    cfg = cfg or Phase3Config()
    inci_tokens = _extract_inci_tokens(routine_filtered)

    # severity = model probability per concern
    severities: Dict[str, float] = {}
    for k, v in (results or {}).items():
        try:
            severities[k] = float((v or {}).get("probability") or 0.0)
        except Exception:
            severities[k] = 0.0

    skin_type = (skin_type_out or {}).get("skin_type")

    benefit: Dict[str, Any] = {}
    for concern in ["acne", "redness", "blackheads", "bags", "hyperpigmentation"]:
        score, drivers = _benefit_score_for_concern(concern, severities.get(concern, 0.0), inci_tokens, cfg)
        benefit[concern] = {
            "score": score,
            "severity": round(severities.get(concern, 0.0), 4),
            "drivers": drivers,
        }

    risk_score, risk_level, risk_drivers = _risk_score(inci_tokens, skin_type, cfg)

    return {
        "outcome": {
            "benefit": benefit,
            "note": "Proxy estimate based on model severity + ingredient presence (not clinical prediction).",
        },
        "risk": {
            "irritation_score": risk_score,
            "level": risk_level,
            "drivers": risk_drivers,
            "note": "Proxy irritation risk based on ingredient flags + skin type (not medical advice).",
        },
    }

