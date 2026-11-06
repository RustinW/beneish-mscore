"""Beneish (1999) 8-variable M-Score: indices, winsorization, scoring, drop log.

Input is a "paired panel": one row per firm-year holding the current-year value
of every line item (`sales`) and the prior-year value (`sales_prev`). Both data
loaders (EDGAR, Compustat CSV) produce this shape, so everything here is
source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

INTERCEPT = -4.84
COEFS = {
    "DSRI": 0.920,
    "GMI": 0.528,
    "AQI": 0.404,
    "SGI": 0.892,
    "DEPI": 0.115,
    "SGAI": -0.172,
    "TATA": 4.679,
    "LVGI": -0.327,
}
INDICES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
# Value of each index for a firm whose ratios did not change (and with zero accruals).
NEUTRAL = {"DSRI": 1.0, "GMI": 1.0, "AQI": 1.0, "SGI": 1.0, "DEPI": 1.0, "SGAI": 1.0, "LVGI": 1.0, "TATA": 0.0}
# M-Score of that "nothing changed" firm: intercept + sum of the ratio coefficients.
M_NEUTRAL = INTERCEPT + sum(COEFS[i] * NEUTRAL[i] for i in INDICES)

# Line items needed for year t and year t-1.
PAIRED_ITEMS = [
    "sales",
    "cogs",
    "receivables",
    "current_assets",
    "ppe",
    "total_assets",
    "depreciation",
    "sga",
    "ltd",
    "current_liabilities",
]
# Line items needed for year t only.
CURRENT_ITEMS = ["net_income", "cfo"]
ID_COLS = ["firm_id", "ticker", "name", "sic", "fiscal_year", "period_end"]
DROP_COLS = ["firm_id", "ticker", "name", "fiscal_year", "stage", "reason"]

# Indices that may optionally be set to their neutral value of 1 instead of
# dropping the firm-year. Off by default; every use is recorded in `imputed`.
NEUTRAL_FILL_ALLOWED = ("GMI", "AQI", "DEPI", "SGAI")


@dataclass
class Config:
    threshold: float = -1.78
    winsor: Tuple[float, float] = (0.01, 0.99)
    winsor_by: str = "year"  # "year" | "pooled" | "none"
    neutral_fill: Tuple[str, ...] = ()  # subset of NEUTRAL_FILL_ALLOWED
    exclude_sic: Tuple[Tuple[int, int], ...] = ((6000, 6999), (4900, 4999))

    def __post_init__(self):
        bad = set(self.neutral_fill) - set(NEUTRAL_FILL_ALLOWED)
        if bad:
            raise ValueError(f"neutral_fill only supports {NEUTRAL_FILL_ALLOWED}, got {sorted(bad)}")
        if self.winsor_by not in ("year", "pooled", "none"):
            raise ValueError("winsor_by must be 'year', 'pooled' or 'none'")
        lo, hi = self.winsor
        if not (0 <= lo < hi <= 1):
            raise ValueError("winsor limits must satisfy 0 <= lower < upper <= 1")


# --------------------------------------------------------------------------- #
# Industry exclusions
# --------------------------------------------------------------------------- #
def sic_exclusion_reason(sic, exclude_sic=Config.exclude_sic) -> str:
    """Return '' if the SIC code is in scope, otherwise why it is excluded."""
    try:
        code = int(float(sic))
    except (TypeError, ValueError):
        return "missing SIC code (cannot confirm firm is not a financial or utility)"
    for lo, hi in exclude_sic:
        if lo <= code <= hi:
            label = {6000: "financial", 4900: "utility"}.get(lo, "excluded industry")
            return f"excluded industry: {label} (SIC {code} in {lo}-{hi})"
    return ""


# --------------------------------------------------------------------------- #
# Index computation. Each function returns (values, reasons): `reasons` is ''
# where the index is valid and a human-readable explanation where it is not.
# --------------------------------------------------------------------------- #
class _Reasons:
    def __init__(self, index):
        self.s = pd.Series("", index=index, dtype=object)

    def add(self, mask, text: str):
        """Record `text` for rows where mask is True and no earlier reason exists."""
        m = mask.fillna(False).astype(bool) & (self.s == "")
        self.s[m] = text

    def missing(self, df, cols):
        for c in cols:
            self.add(df[c].isna(), f"missing {c}")

    def nonpositive(self, df, cols):
        for c in cols:
            self.add(df[c] <= 0, f"nonpositive {c}")

    def negative(self, df, cols):
        for c in cols:
            self.add(df[c] < 0, f"negative {c}")


def _finish(val: pd.Series, r: _Reasons):
    r.add(~np.isfinite(val.astype(float)), "non-finite result")
    return val.where(r.s == "").astype(float), r.s


def _dsri(df):
    r = _Reasons(df.index)
    r.missing(df, ["receivables", "receivables_prev", "sales", "sales_prev"])
    r.nonpositive(df, ["sales", "sales_prev"])
    r.negative(df, ["receivables", "receivables_prev"])
    r.add(df["receivables_prev"] == 0, "zero denominator (receivables_prev = 0)")
    val = (df["receivables"] / df["sales"]) / (df["receivables_prev"] / df["sales_prev"])
    return _finish(val, r)


def _gmi(df):
    r = _Reasons(df.index)
    r.missing(df, ["sales", "sales_prev", "cogs", "cogs_prev"])
    r.nonpositive(df, ["sales", "sales_prev"])
    gm = (df["sales"] - df["cogs"]) / df["sales"]
    gm_prev = (df["sales_prev"] - df["cogs_prev"]) / df["sales_prev"]
    # A zero margin in year t is a zero denominator; a negative margin in either
    # year flips the sign of the ratio and makes the index uninterpretable.
    r.add(gm == 0, "zero denominator (gross margin = 0)")
    r.add(gm < 0, "negative gross margin (index not interpretable)")
    r.add(gm_prev <= 0, "nonpositive gross margin in prior year (index not interpretable)")
    return _finish(gm_prev / gm, r)


def _aqi(df):
    r = _Reasons(df.index)
    cols = ["current_assets", "ppe", "total_assets", "current_assets_prev", "ppe_prev", "total_assets_prev"]
    r.missing(df, cols)
    r.nonpositive(df, ["total_assets", "total_assets_prev"])
    aq = 1 - (df["current_assets"] + df["ppe"]) / df["total_assets"]
    aq_prev = 1 - (df["current_assets_prev"] + df["ppe_prev"]) / df["total_assets_prev"]
    r.add(aq < 0, "current assets + PP&E exceed total assets")
    r.add(aq_prev < 0, "current assets + PP&E exceed total assets in prior year")
    r.add(aq_prev == 0, "zero denominator (no assets outside current assets and PP&E in prior year)")
    return _finish(aq / aq_prev, r)


def _sgi(df):
    r = _Reasons(df.index)
    r.missing(df, ["sales", "sales_prev"])
    r.nonpositive(df, ["sales", "sales_prev"])
    return _finish(df["sales"] / df["sales_prev"], r)


def _depi(df):
    r = _Reasons(df.index)
    r.missing(df, ["depreciation", "depreciation_prev", "ppe", "ppe_prev"])
    r.negative(df, ["depreciation_prev", "ppe", "ppe_prev"])
    r.add(df["depreciation"] <= 0, "zero denominator (depreciation <= 0)")
    r.add((df["depreciation_prev"] + df["ppe_prev"]) <= 0, "zero denominator (depreciation_prev + ppe_prev = 0)")
    rate = df["depreciation"] / (df["depreciation"] + df["ppe"])
    rate_prev = df["depreciation_prev"] / (df["depreciation_prev"] + df["ppe_prev"])
    return _finish(rate_prev / rate, r)


def _sgai(df):
    r = _Reasons(df.index)
    r.missing(df, ["sga", "sga_prev", "sales", "sales_prev"])
    r.nonpositive(df, ["sales", "sales_prev"])
    r.negative(df, ["sga"])
    r.add(df["sga_prev"] <= 0, "zero denominator (sga_prev <= 0)")
    val = (df["sga"] / df["sales"]) / (df["sga_prev"] / df["sales_prev"])
    return _finish(val, r)


def _lvgi(df):
    r = _Reasons(df.index)
    cols = ["ltd", "current_liabilities", "total_assets", "ltd_prev", "current_liabilities_prev", "total_assets_prev"]
    r.missing(df, cols)
    r.nonpositive(df, ["total_assets", "total_assets_prev"])
    r.negative(df, ["ltd", "ltd_prev", "current_liabilities", "current_liabilities_prev"])
    lev = (df["ltd"] + df["current_liabilities"]) / df["total_assets"]
    lev_prev = (df["ltd_prev"] + df["current_liabilities_prev"]) / df["total_assets_prev"]
    r.add(lev_prev == 0, "zero denominator (no debt or current liabilities in prior year)")
    return _finish(lev / lev_prev, r)


def _tata(df):
    r = _Reasons(df.index)
    r.missing(df, ["net_income", "cfo", "total_assets"])
    r.nonpositive(df, ["total_assets"])
    return _finish((df["net_income"] - df["cfo"]) / df["total_assets"], r)


_INDEX_FUNCS = {
    "DSRI": _dsri,
    "GMI": _gmi,
    "AQI": _aqi,
    "SGI": _sgi,
    "DEPI": _depi,
    "SGAI": _sgai,
    "LVGI": _lvgi,
    "TATA": _tata,
}


def required_columns():
    cols = []
    for item in PAIRED_ITEMS:
        cols += [item, item + "_prev"]
    return cols + CURRENT_ITEMS


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().reset_index(drop=True)
    for c in ID_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in required_columns():
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _drops(df: pd.DataFrame, reasons: pd.Series, stage: str) -> pd.DataFrame:
    m = reasons != ""
    out = df.loc[m, ["firm_id", "ticker", "name", "fiscal_year"]].copy()
    out["stage"] = stage
    out["reason"] = reasons[m]
    return out[DROP_COLS]


def winsorize(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clip each index at the configured percentiles, within year or pooled."""
    if cfg.winsor_by == "none" or scored.empty:
        return scored
    lo, hi = cfg.winsor

    def clip(s: pd.Series) -> pd.Series:
        return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))

    out = scored.copy()
    for idx in INDICES:
        if cfg.winsor_by == "year":
            out[idx] = out.groupby("fiscal_year")[idx].transform(clip)
        else:
            out[idx] = clip(out[idx])
    return out


def m_score(df: pd.DataFrame, suffix: str = "") -> pd.Series:
    total = pd.Series(INTERCEPT, index=df.index, dtype=float)
    for idx in INDICES:
        total = total + COEFS[idx] * df[idx + suffix]
    return total


def score_panel(panel: pd.DataFrame, cfg: Optional[Config] = None):
    """Score a paired panel. Returns (scores, drops).

    scores: one row per scored firm-year, sorted by M-Score descending.
    drops:  one row per firm-year removed, with the stage and the reason.
    """
    cfg = cfg or Config()
    df = _prepare(panel)
    drops = []

    # 1. Industry exclusions.
    excl = df["sic"].map(lambda s: sic_exclusion_reason(s, cfg.exclude_sic))
    drops.append(_drops(df, excl, "industry_exclusion"))
    df = df[excl == ""].reset_index(drop=True)

    # 2. Indices, with explicit handling of missing inputs and zero denominators.
    with np.errstate(divide="ignore", invalid="ignore"):
        results = {idx: fn(df) for idx, fn in _INDEX_FUNCS.items()}
    fatal = pd.Series("", index=df.index, dtype=object)
    imputed = pd.Series("", index=df.index, dtype=object)
    for idx in INDICES:
        val, why = results[idx]
        bad = why != ""
        if idx in cfg.neutral_fill:
            val = val.where(~bad, NEUTRAL[idx])
            imputed[bad] = imputed[bad] + (idx + "=1 (" + why[bad] + "); ")
        else:
            fatal[bad] = fatal[bad] + (idx + ": " + why[bad] + "; ")
        df[idx + "_raw"] = val
    fatal = fatal.str.rstrip("; ")
    df["imputed"] = imputed.str.rstrip("; ")
    drops.append(_drops(df, fatal, "index_computation"))
    df = df[fatal == ""].reset_index(drop=True)

    # 3. Winsorize, then score.
    for idx in INDICES:
        df[idx] = df[idx + "_raw"]
    df = winsorize(df, cfg)
    df["m_score"] = m_score(df)
    df["m_score_unwinsorized"] = m_score(df, "_raw")
    df["flag"] = df["m_score"] > cfg.threshold
    df["threshold"] = cfg.threshold

    # Contribution of each index relative to a "nothing changed" firm, so that
    # m_score = M_NEUTRAL + sum(contributions). Shows which component drove the score.
    contrib_cols = []
    for idx in INDICES:
        col = "contrib_" + idx
        df[col] = COEFS[idx] * (df[idx] - NEUTRAL[idx])
        contrib_cols.append(col)
    if len(df):
        df["top_driver"] = df[contrib_cols].idxmax(axis=1).str.replace("contrib_", "", regex=False)
    else:
        df["top_driver"] = pd.Series(dtype=object)

    extra = [c for c in ("notes", "tags_used", "source_ref") if c in df.columns]
    cols = (
        ID_COLS
        + ["m_score", "flag", "threshold", "top_driver"]
        + INDICES
        + contrib_cols
        + [i + "_raw" for i in INDICES]
        + ["m_score_unwinsorized", "imputed"]
        + extra
    )
    scores = df[cols].sort_values("m_score", ascending=False).reset_index(drop=True)
    drop_log = pd.concat(drops, ignore_index=True) if drops else pd.DataFrame(columns=DROP_COLS)
    return scores, drop_log


def flagged_table(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores[scores["flag"]].sort_values("m_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
