"""Compustat loader: a CSV export of comp.funda -> paired firm-year panel.

Expected columns (case-insensitive): gvkey, fyear, datadate, sale, cogs, rect,
act, ppent, at, dp, xsga, dltt, lct, ib (or ibc), oancf, and sich (or sic).
Optional: tic, conm, am, and the usual format filters indfmt/datafmt/consol/popsrc.
"""
from __future__ import annotations

import pandas as pd

from .model import CURRENT_ITEMS, DROP_COLS, PAIRED_ITEMS

# Model line item -> Compustat mnemonic.
MAPPING = {
    "sales": "sale",
    "cogs": "cogs",
    "receivables": "rect",
    "current_assets": "act",
    "ppe": "ppent",
    "total_assets": "at",
    "depreciation": "dp",  # less `am` (amortization of intangibles) when present
    "sga": "xsga",
    "ltd": "dltt",
    "current_liabilities": "lct",
    "net_income": "ib",  # falls back to ibc
    "cfo": "oancf",
}
STANDARD_FILTERS = {"indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D"}


def build_panel(csv_path, start_year: int, end_year: int):
    """Read a funda export and return (panel, drops)."""
    raw = pd.read_csv(csv_path, dtype={"gvkey": str, "GVKEY": str}, low_memory=False)
    raw.columns = [c.lower() for c in raw.columns]
    for col, val in STANDARD_FILTERS.items():
        if col in raw.columns:
            raw = raw[raw[col].astype(str).str.upper() == val]

    need = {"gvkey", "fyear", "datadate"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Compustat file is missing required columns: {sorted(missing)}")

    df = pd.DataFrame({
        "firm_id": raw["gvkey"].astype(str),
        "ticker": raw["tic"] if "tic" in raw.columns else None,
        "name": raw["conm"] if "conm" in raw.columns else None,
        "sic": raw["sich"] if "sich" in raw.columns else None,
        "fiscal_year": pd.to_numeric(raw["fyear"], errors="coerce"),
        "period_end": pd.to_datetime(raw["datadate"].astype(str), errors="coerce"),
    })
    if "sic" in raw.columns:  # header SIC fills gaps in historical SIC
        df["sic"] = df["sic"].fillna(raw["sic"]) if "sich" in raw.columns else raw["sic"]
    for item, col in MAPPING.items():
        df[item] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else float("nan")
    if "ibc" in raw.columns:
        df["net_income"] = df["net_income"].fillna(pd.to_numeric(raw["ibc"], errors="coerce"))
    if "am" in raw.columns:  # Beneish: depreciation = DP less amortization of intangibles
        df["depreciation"] = df["depreciation"] - pd.to_numeric(raw["am"], errors="coerce").fillna(0)

    df = df.dropna(subset=["fiscal_year", "period_end"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df = df.sort_values(["firm_id", "fiscal_year", "period_end"]).drop_duplicates(["firm_id", "fiscal_year"], keep="last")

    # Pair each firm-year with the same firm's immediately preceding fiscal year.
    g = df.groupby("firm_id")
    prev_year = g["fiscal_year"].shift(1)
    gap_days = (df["period_end"] - g["period_end"].shift(1)).dt.days
    consecutive = (prev_year == df["fiscal_year"] - 1) & gap_days.between(350, 380)
    for item in PAIRED_ITEMS:
        df[item + "_prev"] = g[item].shift(1).where(consecutive)

    target = df[df["fiscal_year"].between(start_year, end_year)].copy()
    ok = consecutive.loc[target.index]
    drops = target.loc[~ok, ["firm_id", "ticker", "name", "fiscal_year"]].copy()
    drops["stage"] = "data"
    drops["reason"] = "no consecutive prior fiscal year (12 months earlier) in the Compustat file"
    panel = target[ok].copy()
    panel["period_end"] = panel["period_end"].dt.strftime("%Y-%m-%d")
    panel["source_ref"] = "compustat funda"
    cols = ["firm_id", "ticker", "name", "sic", "fiscal_year", "period_end"]
    for item in PAIRED_ITEMS:
        cols += [item, item + "_prev"]
    cols += CURRENT_ITEMS + ["source_ref"]
    return panel[cols].reset_index(drop=True), drops[DROP_COLS].reset_index(drop=True)
