"""One-page HTML drill-down for a single ticker: 8 indices vs industry median, 3-year trend.

The page is self-contained (inline SVG, no JavaScript or external assets) and
prints on a single sheet.
"""
from __future__ import annotations

import html
import math
from typing import List, Optional

import pandas as pd

from .model import COEFS, INDICES, M_NEUTRAL, NEUTRAL

DESCRIPTIONS = {
    "DSRI": ("Days sales in receivables", "Above 1: receivables grew faster than sales"),
    "GMI": ("Gross margin", "Above 1: gross margin deteriorated"),
    "AQI": ("Asset quality", "Above 1: more assets outside current assets and PP&E"),
    "SGI": ("Sales growth", "Above 1: sales grew"),
    "DEPI": ("Depreciation", "Above 1: depreciation rate slowed"),
    "SGAI": ("SG&A expense", "Above 1: SG&A grew faster than sales"),
    "LVGI": ("Leverage", "Above 1: leverage increased"),
    "TATA": ("Total accruals to total assets", "Higher: earnings less backed by cash"),
}


def _fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}".replace("-", "−")


def _signed(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return ("+" if v >= 0 else "−") + f"{abs(v):.{digits}f}"


def _digits(metric: str) -> int:
    return 3 if metric == "TATA" else 2


def _nice_ticks(lo: float, hi: float, target: int = 3) -> List[float]:
    span = hi - lo
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 5, 10) if s * mag >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    ticks, t = [], first
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _panel(metric: str, title: str, years: List[int], firm: List[float], median: List[float],
           threshold: Optional[float] = None) -> str:
    """One small-multiple line chart: firm vs industry median over the window."""
    W, H, L, R, T, B = 248, 128, 36, 46, 10, 22
    pw, ph = W - L - R, H - T - B
    digits = _digits(metric)
    vals = [v for v in firm + median if v == v]
    if threshold is not None:
        vals.append(threshold)
    if not vals:
        return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: no data"></svg>'
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.1, hi + 0.1
    pad = (hi - lo) * 0.15
    lo, hi = lo - pad, hi + pad

    def x(i):
        return L + (pw * i / (len(years) - 1) if len(years) > 1 else pw / 2)

    def y(v):
        return T + ph * (1 - (v - lo) / (hi - lo))

    def path(series):
        segs, pen = [], False
        for i, v in enumerate(series):
            if v != v:
                pen = False
                continue
            segs.append(f"{'L' if pen else 'M'}{x(i):.1f},{y(v):.1f}")
            pen = True
        return " ".join(segs)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)} trend, firm versus industry median">']
    tick_digits = max(0, min(3, -math.floor(math.log10((hi - lo) / 3)) if hi > lo else 2))
    for t in _nice_ticks(lo, hi):
        p.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        p.append(f'<text class="tick" x="{L - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{_fmt(t, tick_digits)}</text>')
    for i, yr in enumerate(years):
        p.append(f'<text class="tick" x="{x(i):.1f}" y="{H - 6}" text-anchor="middle">FY{str(yr)[2:]}</text>')
    if threshold is not None:
        p.append(f'<line class="thresh" x1="{L}" x2="{W - R}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}"/>')
        p.append(f'<text class="tick" x="{W - R + 6}" y="{y(threshold) + 3.5:.1f}">flag line</text>')
    p.append(f'<path class="median" d="{path(median)}"/>')
    p.append(f'<path class="firm" d="{path(firm)}"/>')
    for i, v in enumerate(firm):
        if v == v:
            p.append(f'<circle class="dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4"/>')
    last = next((i for i in range(len(firm) - 1, -1, -1) if firm[i] == firm[i]), None)
    if last is not None:
        ly = y(firm[last]) + 4
        if threshold is not None and abs(ly - (y(threshold) + 3.5)) < 11 and last == len(years) - 1:
            ly = y(threshold) + (-9 if firm[last] > threshold else 16)
        p.append(f'<text class="endlabel" x="{x(last) + 8:.1f}" y="{ly:.1f}">{_fmt(firm[last], digits)}</text>')
    # Hover targets (larger than the marks) with native tooltips.
    for i, yr in enumerate(years):
        tip = f"FY{yr}: firm {_fmt(firm[i], digits)}, industry median {_fmt(median[i], digits)}"
        p.append(f'<rect class="hit" x="{x(i) - pw / 4:.1f}" y="{T}" width="{pw / 2:.1f}" height="{ph}"><title>{tip}</title></rect>')
    p.append("</svg>")
    return "".join(p)


def peer_medians(scores: pd.DataFrame, sic, years: List[int], min_peers: int = 5):
    """Median of each index among SIC-2 peers per year; falls back to all scored firms."""
    cols = INDICES + ["m_score"]
    sic_num = pd.to_numeric(scores["sic"], errors="coerce")
    try:
        sic2 = int(float(sic)) // 100
    except (TypeError, ValueError):
        sic2 = None
    rows, labels = [], []
    for yr in years:
        in_year = scores["fiscal_year"] == yr
        peers = scores[in_year & (sic_num // 100 == sic2)] if sic2 is not None else scores.iloc[0:0]
        if len(peers) >= min_peers:
            labels.append(f"FY{yr}: {len(peers)} firms in SIC {sic2:02d}xx")
        else:
            n_sic = len(peers)
            peers = scores[in_year]
            labels.append(f"FY{yr}: all {len(peers)} scored firms (only {n_sic} in SIC {sic2 if sic2 is not None else '?'}xx)")
        rows.append(peers[cols].median() if len(peers) else pd.Series(float("nan"), index=cols))
    return pd.DataFrame(rows, index=years), labels


CSS = """
:root{color-scheme:light dark}
.viz-root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--firm:#2a78d6;--median:#898781;--critical:#d03b3b;--good:#0ca30c;--border:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){.viz-root{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--firm:#3987e5;--border:rgba(255,255,255,.10)}}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:20px;min-height:100vh}
.sheet{max-width:820px;margin:0 auto;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px 24px}
h1{font-size:19px;margin:0}
.sub{color:var(--ink2);margin:2px 0 14px}
.hero{display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap;margin-bottom:12px}
.hero .num{font-size:48px;font-weight:600;line-height:1}
.hero .lab{color:var(--ink2);font-size:12px}
.status{font-weight:600}
.status .icon{margin-right:5px}
.status.flagged .icon{color:var(--critical)}
.status.clear .icon{color:var(--good)}
.legend{display:flex;gap:18px;color:var(--ink2);font-size:12px;margin:10px 0 4px;align-items:center}
.key{display:inline-block;width:22px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.key.firm{border-color:var(--firm)}.key.median{border-color:var(--median)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 14px}
.cell h3{font-size:12px;margin:6px 0 0;font-weight:600}
.cell h3 span{font-weight:400;color:var(--ink2)}
svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .thresh{stroke:var(--axis);stroke-width:1}
svg .tick{fill:var(--muted);font-size:9.5px;font-variant-numeric:tabular-nums}
svg .median{fill:none;stroke:var(--median);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .firm{fill:none;stroke:var(--firm);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .dot{fill:var(--firm);stroke:var(--surface);stroke-width:2}
svg .endlabel{fill:var(--ink);font-size:10.5px;font-weight:600}
svg .hit{fill:transparent}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:12px}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td .hint{color:var(--ink2)}
tr.driver td:first-child{font-weight:700}
.notes{color:var(--ink2);font-size:11px;margin-top:10px}
.notes p{margin:3px 0}
details{margin-top:8px;font-size:12px}
summary{cursor:pointer;color:var(--ink2)}
@page{size:letter portrait;margin:10mm}
@media print{.viz-root{background:none;padding:0;font-size:11px}.sheet{border:none;max-width:none;padding:0}
.hero .num{font-size:40px}details{display:none}.cell,table{break-inside:avoid}}
"""


def render(scores: pd.DataFrame, ticker: str, year: Optional[int] = None, min_peers: int = 5) -> str:
    """Return the drill-down page for `ticker` as an HTML string."""
    scores = scores.copy()
    scores["fiscal_year"] = scores["fiscal_year"].astype(int)
    firm_rows = scores[scores["ticker"].astype(str).str.upper() == ticker.upper()]
    if firm_rows.empty:
        raise ValueError(f"{ticker} has no scored firm-years. Check dropped.csv for the reason.")
    year = int(year) if year is not None else int(firm_rows["fiscal_year"].max())
    if year not in set(firm_rows["fiscal_year"]):
        raise ValueError(f"{ticker} was not scored for fiscal year {year}. Scored years: {sorted(firm_rows['fiscal_year'])}")
    years = [year - 2, year - 1, year]
    by_year = firm_rows.drop_duplicates("fiscal_year").set_index("fiscal_year")
    cur = by_year.loc[year]
    med, peer_labels = peer_medians(scores, cur["sic"], years, min_peers)

    def series(col):
        return [float(by_year.loc[y, col]) if y in by_year.index else float("nan") for y in years]

    threshold = float(cur["threshold"])
    flagged = bool(cur["m_score"] > threshold)
    e = html.escape
    name = e(str(cur["name"])) if pd.notna(cur["name"]) else e(ticker.upper())

    cells = [
        f'<div class="cell"><h3>M-Score</h3>'
        + _panel("M", "M-Score", years, series("m_score"), [float(v) for v in med["m_score"]], threshold)
        + "</div>"
    ]
    for idx in INDICES:
        cells.append(
            f'<div class="cell"><h3>{idx} <span>{e(DESCRIPTIONS[idx][0])}</span></h3>'
            + _panel(idx, idx, years, series(idx), [float(v) for v in med[idx]])
            + "</div>"
        )

    rows = []
    for idx in INDICES:
        d = _digits(idx)
        fv, mv = float(cur[idx]), float(med.loc[year, idx])
        cls = ' class="driver"' if idx == cur["top_driver"] else ""
        rows.append(
            f"<tr{cls}><td>{idx}</td><td>{e(DESCRIPTIONS[idx][0])}<br><span class=\"hint\">{e(DESCRIPTIONS[idx][1])}</span></td>"
            f'<td class="n">{_fmt(fv, d)}</td><td class="n">{_fmt(mv, d)}</td><td class="n">{_signed(fv - mv, d)}</td>'
            f'<td class="n">{_signed(COEFS[idx] * (fv - NEUTRAL[idx]))}</td></tr>'
        )

    hist = []
    for col in ["m_score"] + INDICES:
        d = _digits(col)
        f_cells = "".join(f'<td class="n">{_fmt(v, d)}</td>' for v in series(col))
        m_cells = "".join(f'<td class="n">{_fmt(float(v), d)}</td>' for v in med[col])
        hist.append(f"<tr><td>{'M-Score' if col == 'm_score' else col}</td>{f_cells}{m_cells}</tr>")
    yr_heads = "".join(f'<th class="n">FY{y}</th>' for y in years)

    missing_years = [y for y in years if y not in by_year.index]
    notes = [f"Industry median: {'; '.join(peer_labels)}."]
    if missing_years:
        notes.append("No score for " + ", ".join(f"FY{y}" for y in missing_years) + " (not in the run or dropped; see dropped.csv).")
    notes.append(
        "Indices are shown after winsorization, as used in the score. Contribution = coefficient × (index − neutral value), "
        f"where neutral is 1 (0 for TATA); M-Score = {_fmt(M_NEUTRAL)} + sum of contributions."
    )
    for label, col in (("Imputed", "imputed"), ("Data notes", "notes"), ("Source", "source_ref")):
        if col in cur.index and pd.notna(cur[col]) and str(cur[col]).strip():
            notes.append(f"{label}: {e(str(cur[col]))}.")
    notes.append("A high M-Score is a statistical screen, not evidence of manipulation.")

    status = (
        f'<span class="status flagged"><span class="icon">▲</span>Flagged: above {_fmt(threshold)}</span>'
        if flagged else
        f'<span class="status clear"><span class="icon">●</span>Not flagged: at or below {_fmt(threshold)}</span>'
    )
    top = cur["top_driver"]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(ticker.upper())} Beneish M-Score drill-down FY{year}</title><style>{CSS}</style></head>
<body><div class="viz-root"><div class="sheet">
<h1>{name} ({e(ticker.upper())})</h1>
<div class="sub">Beneish M-Score drill-down &middot; fiscal year {year} (period ending {e(str(cur['period_end']))}) &middot; SIC {e(str(cur['sic']))}</div>
<div class="hero">
  <div><div class="lab">M-Score, FY{year}</div><div class="num">{_fmt(float(cur['m_score']))}</div></div>
  <div><div class="lab">Screen result</div><div>{status}</div></div>
  <div><div class="lab">Largest contributor</div><div><b>{top}</b> ({_signed(float(cur['contrib_' + top]))} to the score)</div></div>
  <div><div class="lab">Industry median M-Score</div><div>{_fmt(float(med.loc[year, 'm_score']))}</div></div>
</div>
<div class="legend"><span><span class="key firm"></span>{e(ticker.upper())}</span><span><span class="key median"></span>Industry median</span></div>
<div class="grid3">{''.join(cells)}</div>
<table><thead><tr><th>Index</th><th>What it measures</th><th class="n">{e(ticker.upper())}</th><th class="n">Industry median</th>
<th class="n">Difference</th><th class="n">Contribution to M</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<details><summary>Three-year data table</summary>
<table><thead><tr><th></th><th colspan="3">{e(ticker.upper())}</th><th colspan="3">Industry median</th></tr>
<tr><th>Metric</th>{yr_heads}{yr_heads}</tr></thead><tbody>{''.join(hist)}</tbody></table></details>
<div class="notes">{''.join(f'<p>{n}</p>' for n in notes)}</div>
</div></div></body></html>
"""
