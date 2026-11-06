"""Command line: python -m beneish {run,validate,drilldown} ..."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import compustat, drilldown, edgar, validate
from .model import DROP_COLS, NEUTRAL_FILL_ALLOWED, Config, flagged_table, score_panel

ID_DTYPES = {"firm_id": str, "ticker": str, "sic": str}


def read_scores(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=ID_DTYPES)


def read_universe(path) -> list:
    """One ticker or CIK per line; a CSV works too (first column, header optional)."""
    out = []
    for line in Path(path).read_text().splitlines():
        token = line.split(",")[0].strip().strip('"')
        if token and not token.startswith("#") and token.lower() not in ("ticker", "cik", "symbol", "firm_id"):
            out.append(token)
    return out


def cmd_run(a) -> int:
    cfg = Config(threshold=a.threshold, winsor=tuple(a.winsor), winsor_by=a.winsor_by, neutral_fill=tuple(a.neutral_fill))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.source == "edgar":
        if not a.universe:
            sys.exit("--universe FILE is required for --source edgar (one ticker or CIK per line)")
        client = edgar.EdgarClient(a.user_agent or os.environ.get("SEC_USER_AGENT", ""), a.cache, refresh=a.refresh)
        ids = read_universe(a.universe)
        print(f"EDGAR: {len(ids)} identifiers, fiscal years {a.start_year}-{a.end_year}", file=sys.stderr)
        panel, data_drops = edgar.build_panel(client, ids, a.start_year, a.end_year, cfg.exclude_sic)
    else:
        if not a.input:
            sys.exit("--input FILE is required for --source compustat (CSV export of comp.funda)")
        panel, data_drops = compustat.build_panel(a.input, a.start_year, a.end_year)

    panel.to_csv(out / "fundamentals.csv", index=False)
    scores, model_drops = score_panel(panel, cfg)
    drops = pd.concat([data_drops, model_drops], ignore_index=True)[DROP_COLS]
    drops["fiscal_year"] = pd.to_numeric(drops["fiscal_year"], errors="coerce").astype("Int64")  # blank = whole firm
    flagged = flagged_table(scores)
    scores.to_csv(out / "scores.csv", index=False)
    flagged.to_csv(out / "flagged.csv", index=False)
    drops.to_csv(out / "dropped.csv", index=False)

    print(f"Scored firm-years: {len(scores)}  ({scores['firm_id'].nunique()} firms)")
    print(f"Flagged (M > {cfg.threshold}): {len(flagged)}")
    print(f"Dropped entries: {len(drops)}  (firm-level: {int(drops['fiscal_year'].isna().sum())}, firm-year: {int(drops['fiscal_year'].notna().sum())})")
    if len(drops):
        print("Drops by stage:")
        for stage, n in drops["stage"].value_counts().items():
            print(f"  {stage}: {n}")
    print(f"Wrote {out}/scores.csv, flagged.csv, dropped.csv, fundamentals.csv")
    return 0


def cmd_validate(a) -> int:
    scores = read_scores(a.scores)
    drops = pd.read_csv(a.dropped, dtype=ID_DTYPES) if a.dropped and Path(a.dropped).exists() else None
    result = validate.backtest(scores, validate.load_labels(a.labels), drops, a.thresholds)
    report = validate.render_report(result)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    result["table"].to_csv(out.with_suffix(".csv"), index=False)
    print(report)
    print(f"Wrote {out} and {out.with_suffix('.csv')}")
    return 0


def cmd_drilldown(a) -> int:
    scores = read_scores(a.scores)
    page = drilldown.render(scores, a.ticker, a.year, a.min_peers)
    out = Path(a.out) if a.out else Path(a.scores).parent / f"drilldown_{a.ticker.upper()}.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="beneish", description="Beneish M-Score screen")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build the panel, compute indices, score and flag")
    r.add_argument("--source", choices=["edgar", "compustat"], default="edgar")
    r.add_argument("--universe", help="EDGAR: file of tickers or CIKs, one per line")
    r.add_argument("--input", help="Compustat: CSV export of comp.funda")
    r.add_argument("--start-year", type=int, required=True)
    r.add_argument("--end-year", type=int, required=True)
    r.add_argument("--threshold", type=float, default=-1.78, help="flag firm-years with M above this (default -1.78)")
    r.add_argument("--winsor", type=float, nargs=2, default=[0.01, 0.99], metavar=("LOWER", "UPPER"))
    r.add_argument("--winsor-by", choices=["year", "pooled", "none"], default="year")
    r.add_argument("--neutral-fill", nargs="*", default=[], choices=NEUTRAL_FILL_ALLOWED,
                   help="set these indices to 1 instead of dropping the firm-year when they cannot be computed")
    r.add_argument("--out", default="output")
    r.add_argument("--cache", default="cache/edgar")
    r.add_argument("--refresh", action="store_true", help="ignore the EDGAR cache and download again")
    r.add_argument("--user-agent", help="SEC User-Agent, e.g. 'Your Name you@example.com' (or set SEC_USER_AGENT)")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("validate", help="back-test against a labels file of known cases")
    v.add_argument("--labels", required=True, help="CSV: fiscal_year plus firm_id (cik/gvkey) or ticker")
    v.add_argument("--scores", default="output/scores.csv")
    v.add_argument("--dropped", default="output/dropped.csv")
    v.add_argument("--thresholds", type=float, nargs="+", default=[-1.78, -2.22])
    v.add_argument("--out", default="output/validation.md")
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("drilldown", help="one-page HTML report for a ticker")
    d.add_argument("ticker")
    d.add_argument("--year", type=int, help="fiscal year (default: latest scored)")
    d.add_argument("--scores", default="output/scores.csv")
    d.add_argument("--min-peers", type=int, default=5, help="minimum SIC-2 peers before falling back to all firms")
    d.add_argument("--out")
    d.set_defaults(func=cmd_drilldown)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
