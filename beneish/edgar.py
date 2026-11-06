"""SEC EDGAR XBRL loader: company facts API -> paired firm-year panel.

For each fiscal year the current-year AND prior-year values are both read from
the same originally filed 10-K (the comparative columns), so the two years are
on the same accounting basis and use the same XBRL tags.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .model import DROP_COLS, sic_exclusion_reason

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

ANNUAL_FORMS = ("10-K", "10-K/A")
ANNUAL_DAYS = (350, 380)  # accepted length of a fiscal year (covers 52/53-week years)

# us-gaap tags tried in priority order for each line item. A tag that has both
# the current and prior year in the filing is preferred over one that has only one.
TAGS: Dict[str, List[str]] = {
    "sales": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    # cogs is derived as sales - GrossProfit when GrossProfit is tagged; otherwise:
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
    ],
    "current_assets": ["AssetsCurrent"],
    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "total_assets": ["Assets"],
    "depreciation": [
        "Depreciation",
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "ltd": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermNotesPayable",
        "SeniorLongTermNotes",
        "ConvertibleNotesPayableNoncurrent",
        "ConvertibleLongTermNotesPayable",
        "ConvertibleDebtNoncurrent",
        "LongTermLineOfCredit",
        "OtherLongTermDebtNoncurrent",
    ],
    "current_liabilities": ["LiabilitiesCurrent"],
    "net_income": [
        "IncomeLossFromContinuingOperations",
        "ProfitLoss",
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
# Helper tags used by the fallbacks in _resolve().
EXTRA_TAGS = [
    "GrossProfit",
    "GeneralAndAdministrativeExpense",
    "SellingAndMarketingExpense",
    "SellingExpense",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtCurrent",
]
ALL_TAGS = sorted({t for tags in TAGS.values() for t in tags} | set(EXTRA_TAGS))
INSTANT_ITEMS = {"receivables", "current_assets", "ppe", "total_assets", "ltd", "current_liabilities"}


def fiscal_year_label(period_end: str) -> int:
    """Compustat convention: fiscal years ending Jan-May belong to the prior calendar year."""
    d = date.fromisoformat(period_end)
    return d.year if d.month >= 6 else d.year - 1


def pad_cik(cik) -> str:
    return str(int(cik)).zfill(10)


# --------------------------------------------------------------------------- #
# HTTP client with on-disk cache
# --------------------------------------------------------------------------- #
class EdgarClient:
    def __init__(self, user_agent: str, cache_dir="cache/edgar", min_interval: float = 0.15, refresh: bool = False):
        if not user_agent:
            raise ValueError(
                "SEC requires automated clients to identify themselves. Set SEC_USER_AGENT "
                "(or pass --user-agent) to something like 'Your Name your.email@example.com'."
            )
        self.user_agent = user_agent
        self.cache = Path(cache_dir)
        self.min_interval = min_interval
        self.refresh = refresh
        self._last = 0.0

    def _get_json(self, url: str):
        """GET a JSON document. Returns None on 404. Retries on throttling/server errors."""
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                if e.code == 403:
                    raise RuntimeError(
                        f"SEC returned 403 for {url}. The User-Agent was rejected or the client was "
                        "rate-limited. Use a User-Agent of the form 'Your Name your.email@example.com'."
                    ) from e
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
        return None

    def _cached(self, path: Path, build):
        if path.exists() and not self.refresh:
            with gzip.open(path, "rt") as f:
                return json.load(f)
        data = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        return data

    def company_meta(self, cik: str) -> dict:
        """{'found', 'name', 'sic', 'sic_description', 'tickers'} for a CIK."""

        def build():
            raw = self._get_json(SUBMISSIONS_URL.format(cik=cik))
            if raw is None:
                return {"found": False}
            return {
                "found": True,
                "name": raw.get("name"),
                "sic": raw.get("sic") or None,
                "sic_description": raw.get("sicDescription"),
                "tickers": raw.get("tickers") or [],
            }

        return self._cached(self.cache / "meta" / f"{cik}.json.gz", build)

    def company_facts(self, cik: str) -> dict:
        """Annual-report facts for the mapped tags only (the raw file is several MB)."""

        def build():
            raw = self._get_json(FACTS_URL.format(cik=cik))
            return {"found": raw is not None, "facts": trim_facts(raw) if raw else {}}

        return self._cached(self.cache / "facts" / f"{cik}.json.gz", build)

    def ticker_map(self) -> Dict[str, str]:
        """Ticker -> zero-padded CIK, from SEC's company_tickers.json."""

        def build():
            raw = self._get_json(TICKERS_URL)
            if raw is None:
                raise RuntimeError("Could not download SEC ticker list")
            return {v["ticker"].upper(): pad_cik(v["cik_str"]) for v in raw.values()}

        return self._cached(self.cache / "tickers.json.gz", build)


def trim_facts(raw: dict) -> Dict[str, list]:
    out = {}
    usgaap = (raw.get("facts") or {}).get("us-gaap") or {}
    for tag in ALL_TAGS:
        node = usgaap.get(tag)
        if not node:
            continue
        rows = [
            {k: f.get(k) for k in ("start", "end", "val", "accn", "form", "filed")}
            for f in (node.get("units") or {}).get("USD", [])
            if f.get("form") in ANNUAL_FORMS
        ]
        if rows:
            out[tag] = rows
    return out


# --------------------------------------------------------------------------- #
# Facts -> firm-year rows
# --------------------------------------------------------------------------- #
def _index_filings(facts: Dict[str, list]) -> Dict[str, dict]:
    """Group facts by filing (accession number), split into instants and full-year durations."""
    filings: Dict[str, dict] = {}
    for tag, flist in facts.items():
        for f in flist:
            a = filings.setdefault(
                f["accn"], {"accn": f["accn"], "form": f["form"], "filed": f["filed"], "inst": {}, "dur": {}}
            )
            if f.get("start") is None:
                a["inst"].setdefault((tag, f["end"]), f["val"])
            else:
                days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days
                if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
                    a["dur"].setdefault((tag, f["end"]), f["val"])
    return filings


def _pick(store: dict, tags: List[str], end: str, end_prev: str):
    """Return (cur, prev, tag_cur, tag_prev), preferring one tag that covers both years."""
    for t in tags:
        if (t, end) in store and (t, end_prev) in store:
            return store[(t, end)], store[(t, end_prev)], t, t
    cur = next(((store[(t, end)], t) for t in tags if (t, end) in store), (None, None))
    prev = next(((store[(t, end_prev)], t) for t in tags if (t, end_prev) in store), (None, None))
    return cur[0], prev[0], cur[1], prev[1]


def _resolve(filing: dict, end: str, end_prev: str):
    """Map one filing's facts onto the model's line items. Returns (values, tags_used, notes)."""
    inst, dur = filing["inst"], filing["dur"]
    vals: Dict[str, Optional[float]] = {}
    tags: Dict[str, str] = {}
    notes: List[str] = []

    def simple(item):
        store = inst if item in INSTANT_ITEMS else dur
        cur, prev, t_cur, t_prev = _pick(store, TAGS[item], end, end_prev)
        vals[item], vals[item + "_prev"] = cur, prev
        if t_cur or t_prev:
            tags[item] = t_cur if t_cur == t_prev else f"{t_cur}|{t_prev}"
        if t_cur and t_prev and t_cur != t_prev:
            notes.append(f"{item}: different tags for current and prior year")

    # Sales: filers use `Revenues` inconsistently (some tag only a side item with
    # it, or report it as 0 next to SalesRevenueNet), so take the revenue tag with
    # the largest current-year value among those reported for both years.
    both = [t for t in TAGS["sales"] if (t, end) in dur and (t, end_prev) in dur]
    if both:
        best = max(both, key=lambda t: dur[(t, end)])
        vals["sales"], vals["sales_prev"], tags["sales"] = dur[(best, end)], dur[(best, end_prev)], best
    else:
        simple("sales")

    for item in ("receivables", "current_assets", "ppe", "total_assets", "depreciation", "current_liabilities"):
        simple(item)

    # COGS: sales - GrossProfit when the filer tags gross profit (it ties to the
    # reported margin even when cost of sales is split across several tags).
    gp, gp_prev = dur.get(("GrossProfit", end)), dur.get(("GrossProfit", end_prev))
    if gp is not None and gp_prev is not None and vals["sales"] is not None and vals["sales_prev"] is not None:
        vals["cogs"], vals["cogs_prev"] = vals["sales"] - gp, vals["sales_prev"] - gp_prev
        tags["cogs"] = "sales-GrossProfit"
    else:
        simple("cogs")

    # SG&A: the combined tag, else G&A plus selling/marketing expense.
    simple("sga")
    if vals["sga"] is None or vals["sga_prev"] is None:
        ga, ga_prev = dur.get(("GeneralAndAdministrativeExpense", end)), dur.get(("GeneralAndAdministrativeExpense", end_prev))
        if ga is not None and ga_prev is not None:
            sell_tag = next(
                (t for t in ("SellingAndMarketingExpense", "SellingExpense") if (t, end) in dur and (t, end_prev) in dur),
                None,
            )
            sell, sell_prev = (dur[(sell_tag, end)], dur[(sell_tag, end_prev)]) if sell_tag else (0, 0)
            vals["sga"], vals["sga_prev"] = ga + sell, ga_prev + sell_prev
            tags["sga"] = "GeneralAndAdministrativeExpense" + (f"+{sell_tag}" if sell_tag else "")
            notes.append("sga: built from components" if sell_tag else "sga: G&A only (no selling expense tag)")

    # Long-term debt: noncurrent debt tags, else total debt less its current
    # portion, else zero (filers with no debt simply do not report the tag).
    simple("ltd")
    for suffix, d in (("", end), ("_prev", end_prev)):
        key = "ltd" + suffix
        if vals[key] is None:
            for total_tag, current_tag in (("LongTermDebt", "LongTermDebtCurrent"),
                                           ("DebtLongtermAndShorttermCombinedAmount", "DebtCurrent")):
                total = inst.get((total_tag, d))
                if total is not None:
                    vals[key] = max(total - (inst.get((current_tag, d)) or 0), 0)
                    tags[key] = f"{total_tag}-{current_tag}"
                    break
            else:
                vals[key] = 0.0
                notes.append(f"{key}: no long-term debt tag in filing, assumed 0")

    for item in ("net_income", "cfo"):
        hit = next(((dur[(t, end)], t) for t in TAGS[item] if (t, end) in dur), (None, None))
        vals[item] = hit[0]
        if hit[1]:
            tags[item] = hit[1]
    return vals, tags, notes


def firm_year_rows(ident: dict, facts: Dict[str, list], start_year: int, end_year: int) -> Tuple[list, list]:
    """Build paired rows for one firm. `ident` holds firm_id, ticker, name, sic."""
    rows, drops = [], []

    def drop(fy, reason):
        drops.append({"firm_id": ident["firm_id"], "ticker": ident.get("ticker"), "name": ident.get("name"),
                      "fiscal_year": fy, "stage": "data", "reason": reason})

    # Each filing's fiscal year end is its latest balance sheet date.
    by_end: Dict[str, list] = {}
    for a in _index_filings(facts).values():
        ends = [e for (t, e) in a["inst"] if t == "Assets"]
        if ends:
            by_end.setdefault(max(ends), []).append(a)

    by_fy: Dict[int, Tuple[str, dict]] = {}
    for end in sorted(by_end):
        # Original 10-K if there is one, otherwise the earliest amendment.
        filing = sorted(by_end[end], key=lambda a: (a["form"] != "10-K", a["filed"]))[0]
        fy = fiscal_year_label(end)
        if fy in by_fy and start_year <= fy <= end_year:
            drop(fy, f"fiscal year end changed: period ending {by_fy[fy][0]} superseded by {end} for the same fiscal-year label")
        by_fy[fy] = (end, filing)

    for fy in range(start_year, end_year + 1):
        if fy not in by_fy:
            drop(fy, "no 10-K with a tagged balance sheet (us-gaap Assets) for this fiscal year")
            continue
        end, filing = by_fy[fy]
        d_end = date.fromisoformat(end)
        prev_ends = [
            e for (t, e) in filing["inst"]
            if t == "Assets" and ANNUAL_DAYS[0] <= (d_end - date.fromisoformat(e)).days <= ANNUAL_DAYS[1]
        ]
        if not prev_ends:
            drop(fy, "10-K has no prior-year balance sheet 350-380 days earlier (new filer or changed fiscal year end)")
            continue
        vals, tags, notes = _resolve(filing, end, max(prev_ends))
        rows.append({
            **{k: ident.get(k) for k in ("firm_id", "ticker", "name", "sic")},
            "fiscal_year": fy,
            "period_end": end,
            **vals,
            "notes": "; ".join(notes),
            "tags_used": "; ".join(f"{k}={v}" for k, v in tags.items()),
            "source_ref": f"{filing['form']} {filing['accn']} filed {filing['filed']}",
        })
    return rows, drops


def resolve_universe(client: EdgarClient, identifiers: List[str]) -> Tuple[List[Tuple[str, Optional[str]]], list]:
    """Turn tickers / CIKs into [(cik, ticker)]. Unknown tickers go to the drop log."""
    out, drops, seen = [], [], set()
    tmap = None
    for ident in identifiers:
        ident = ident.strip()
        if not ident:
            continue
        if ident.isdigit():
            cik, ticker = pad_cik(ident), None
        else:
            if tmap is None:
                tmap = client.ticker_map()
            ticker = ident.upper()
            cik = tmap.get(ticker) or tmap.get(ticker.replace(".", "-"))
            if cik is None:
                drops.append({"firm_id": None, "ticker": ticker, "name": None, "fiscal_year": None,
                              "stage": "universe", "reason": "ticker not found in SEC company_tickers.json"})
                continue
        if cik not in seen:
            seen.add(cik)
            out.append((cik, ticker))
    return out, drops


def build_panel(client: EdgarClient, identifiers: List[str], start_year: int, end_year: int,
                exclude_sic=((6000, 6999), (4900, 4999)), progress: bool = True):
    """Download (or read from cache) every firm and return (panel, drops)."""
    universe, drops = resolve_universe(client, identifiers)
    rows = []
    for i, (cik, ticker) in enumerate(universe, 1):
        if progress and (i % 25 == 0 or i == len(universe)):
            print(f"  [{i}/{len(universe)}] firms processed", file=sys.stderr, flush=True)
        ident = {"firm_id": cik, "ticker": ticker, "name": None, "sic": None}

        def firm_drop(reason, stage="data"):
            drops.append({"firm_id": cik, "ticker": ident["ticker"], "name": ident["name"],
                          "fiscal_year": None, "stage": stage, "reason": reason})

        try:
            meta = client.company_meta(cik)
            if not meta.get("found"):
                firm_drop("CIK not found in SEC submissions API")
                continue
            ident["name"], ident["sic"] = meta.get("name"), meta.get("sic")
            if not ident["ticker"] and meta.get("tickers"):
                ident["ticker"] = meta["tickers"][0]
            why = sic_exclusion_reason(ident["sic"], exclude_sic)
            if why:
                firm_drop(why, "industry_exclusion")
                continue
            facts = client.company_facts(cik)
        except Exception as e:  # network failure for one firm should not sink the run
            firm_drop(f"download failed: {type(e).__name__}: {e}")
            continue
        if not facts.get("facts"):
            firm_drop("no us-gaap annual (10-K) XBRL facts on EDGAR (pre-XBRL, IFRS/foreign filer, or not an operating company)")
            continue
        r, d = firm_year_rows(ident, facts["facts"], start_year, end_year)
        rows += r
        drops += d
    panel = pd.DataFrame(rows)
    drop_log = pd.DataFrame(drops, columns=DROP_COLS)
    return panel, drop_log
