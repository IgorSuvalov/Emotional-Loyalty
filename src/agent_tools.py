import pandas as pd
from typing import Dict
from dataclasses import dataclass
from src.scoring import run_loyalty_model


@dataclass
class ModelParams:
    spend: float
    engage: float
    lam: float
    multipliers: Dict[str, float]
    tier_mix: Dict[str, float]


def _norm_knobs(spend, engage):
    total = spend + engage
    return {
        "spend": spend / total,
        "engage": engage / total,
    }

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _sanitize(data: dict, defaults: ModelParams):
    spend = float(data.get("spend", defaults.spend))
    engage = float(data.get("engage", defaults.engage))
    spend, engage = _norm_knobs(spend, engage).values()

    lam = _clamp(data.get("lam", defaults.lam), 0.0, 1.0)

    mp_in = data.get("multipliers", {}) or {}
    multipliers = {}
    for k in ["Brand Champions", "Transactional Spenders", "Brand Advocates", "Passive Customers"]:
        v = float(mp_in.get(k, defaults.multipliers[k]))
        multipliers[k] = _clamp(v, 0.80, 1.20)

    mix_in = data.get("tier_mix", {}) or {}
    tier_mix: Dict[str, float] = {}
    for k in ["Platinum", "Gold", "Silver", "Regular"]:
        tier_mix[k] = float(mix_in.get(k, defaults.tier_mix[k]))
    s = sum(tier_mix.values())
    for k in tier_mix:
        tier_mix[k] = tier_mix[k] / s

    return ModelParams(spend=spend, engage=engage, lam=lam, multipliers=multipliers, tier_mix=tier_mix)


def run_model(df: pd.DataFrame, params: ModelParams):
    knobs = _norm_knobs(params.spend, params.engage)
    return run_loyalty_model(df, knobs, params.lam, params.multipliers, params.tier_mix)


def summarize_results(final_df: pd.DataFrame):
    mix = final_df['tier'].value_counts(normalize=True).to_dict()
    return {"tier_distribution": mix, "avg_score": final_df['score'].mean()}
