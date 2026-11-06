"""Back-test scored firm-years against a user-supplied list of known manipulation cases.

Nothing here ships with labels: the hit rate and false positive rate are only
ever computed from the labels file you provide.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import pandas as pd


def _norm_id(s: pd.Series) -> pd.Series:
    """Normalize firm ids so '320193', 320193 and '0000320193' all match."""
    s = s.astype(str).str.strip()
    return s.where(~s.str.fullmatch(r"\d+"), s.str.lstrip("0"))


def load_labels(path) -> pd.DataFrame:
    """Labels CSV: one row per known manipulation firm-year.

    Required: fiscal_year, plus firm_id (CIK / gvkey) and/or ticker.
    """
    lab = pd.read_csv(path, dtype=str, comment="#")
    lab.columns = [c.strip().lower() for c in lab.columns]
    if "cik" in lab.columns and "firm_id" not in lab.columns:
        lab = lab.rename(columns={"cik": "firm_id"})
    if "gvkey" in lab.columns and "firm_id" not in lab.columns:
        lab = lab.rename(columns={"gvkey": "firm_id"})
    if "fiscal_year" not in lab.columns or not ({"firm_id", "ticker"} & set(lab.columns)):
        raise ValueError("labels file needs a fiscal_year column and a firm_id (cik/gvkey) or ticker column")
    lab["fiscal_year"] = pd.to_numeric(lab["fiscal_year"], errors="coerce")
    lab = lab.dropna(subset=["fiscal_year"])
    lab["fiscal_year"] = lab["fiscal_year"].astype(int)
    return lab.reset_index(drop=True)


def _keys(df: pd.DataFrame, col: str) -> pd.Series:
    ident = _norm_id(df[col]) if col == "firm_id" else df[col].astype(str).str.strip().str.upper()
    key = ident + "|" + df["fiscal_year"].astype(int).astype(str)
    return key.where(df[col].notna() & (df[col].astype(str).str.strip() != ""))


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def backtest(scores: pd.DataFrame, labels: pd.DataFrame, drops: Optional[pd.DataFrame] = None,
             thresholds: Iterable[float] = (-1.78, -2.22)) -> dict:
    """Match labels to scored firm-years and compute hit / false-positive rates."""
    scores = scores.copy()
    scores["known_case"] = False
    labels = labels.copy()
    labels["status"] = "not in scored output or drop log (outside universe or year range)"

    # Match on firm_id when the label has one, otherwise on ticker.
    for col in ("firm_id", "ticker"):
        if col not in labels.columns or col not in scores.columns:
            continue
        lab_key = _keys(labels, col)
        score_key = _keys(scores, col)
        hit = lab_key.isin(set(score_key.dropna()))
        labels.loc[hit, "status"] = "scored"
        scores.loc[score_key.isin(set(lab_key.dropna())), "known_case"] = True
        if drops is not None and len(drops):
            d = drops.dropna(subset=["fiscal_year"]).copy()
            if len(d):
                d["key"] = _keys(d, col)
                reason = d.dropna(subset=["key"]).groupby("key")["reason"].first()
                dropped = lab_key.isin(reason.index) & (labels["status"] != "scored")
                labels.loc[dropped, "status"] = "dropped: " + lab_key[dropped].map(reason)

    pos = scores[scores["known_case"]]
    neg = scores[~scores["known_case"]]
    rows = []
    for t in thresholds:
        tp = int((pos["m_score"] > t).sum())
        fn = len(pos) - tp
        fp = int((neg["m_score"] > t).sum())
        tn = len(neg) - fp
        lo, hi = wilson_interval(tp, len(pos))
        rows.append({
            "threshold": t,
            "known_cases_scored": len(pos),
            "hits": tp,
            "misses": fn,
            "hit_rate": tp / len(pos) if len(pos) else float("nan"),
            "hit_rate_ci95_low": lo,
            "hit_rate_ci95_high": hi,
            "other_firm_years": len(neg),
            "false_positives": fp,
            "false_positive_rate": fp / len(neg) if len(neg) else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        })
    return {
        "labels_total": len(labels),
        "labels_scored": int((labels["status"] == "scored").sum()),
        "labels": labels,
        "table": pd.DataFrame(rows),
        "cases": pos.sort_values("m_score", ascending=False),
    }


def _pct(x) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def render_report(result: dict) -> str:
    """Markdown report. Every number is computed from the supplied labels and scores."""
    lab, table = result["labels"], result["table"]
    out = ["# Beneish M-Score back-test", ""]
    out.append(f"- Known manipulation firm-years in labels file: **{result['labels_total']}**")
    out.append(f"- Of those, scored by the model: **{result['labels_scored']}**")
    unscored = lab[lab["status"] != "scored"]
    if len(unscored):
        out.append(f"- Not scored: **{len(unscored)}** (listed below; these are excluded from the hit rate)")
    out.append("")
    if result["labels_scored"] == 0:
        out.append("**No labelled firm-year matched a scored firm-year, so no hit rate can be reported.**")
        out.append("Check that the labels use the same identifiers (CIK/gvkey or ticker) and fiscal-year convention as the scores file.")
    else:
        out.append("| Threshold | Hit rate | 95% CI | Hits / known cases | False positive rate | False positives / other firm-years | Precision |")
        out.append("|---|---|---|---|---|---|---|")
        for r in table.itertuples():
            out.append(
                f"| M > {r.threshold} | {_pct(r.hit_rate)} | {_pct(r.hit_rate_ci95_low)}-{_pct(r.hit_rate_ci95_high)} "
                f"| {r.hits} / {r.known_cases_scored} | {_pct(r.false_positive_rate)} "
                f"| {r.false_positives} / {r.other_firm_years} | {_pct(r.precision)} |"
            )
        out += [
            "",
            "Hit rate = known cases with M above the threshold / known cases scored. "
            "False positive rate = unlabelled firm-years above the threshold / unlabelled firm-years scored. "
            "Unlabelled does not mean clean: undetected or unlisted manipulators count as false positives here, "
            "so the false positive rate is an upper bound. The confidence interval is a Wilson interval.",
            "",
            "## Known cases scored",
            "",
            "| Ticker | Firm id | Fiscal year | M-Score | Top driver |",
            "|---|---|---|---|---|",
        ]
        for r in result["cases"].itertuples():
            out.append(f"| {r.ticker} | {r.firm_id} | {r.fiscal_year} | {r.m_score:.2f} | {r.top_driver} |")
    if len(unscored):
        out += ["", "## Known cases not scored", "", "| Ticker | Firm id | Fiscal year | Why |", "|---|---|---|---|"]
        for _, r in unscored.iterrows():
            out.append(f"| {r.get('ticker', '')} | {r.get('firm_id', '')} | {r['fiscal_year']} | {r['status']} |")
    return "\n".join(out) + "\n"
