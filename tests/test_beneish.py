"""Unit tests. All data here is synthetic and exists only to check the arithmetic."""
import math

import numpy as np
import pandas as pd
import pytest

from beneish import compustat, drilldown, edgar, validate
from beneish.model import COEFS, INDICES, M_NEUTRAL, Config, flagged_table, score_panel

# One firm-year worked by hand (see expected values below).
BASE = dict(
    firm_id="1", ticker="AAA", name="Alpha Co", sic="3570", fiscal_year=2022, period_end="2022-12-31",
    sales=1200, sales_prev=1000, cogs=780, cogs_prev=600, receivables=150, receivables_prev=100,
    current_assets=500, current_assets_prev=400, ppe=330, ppe_prev=300, total_assets=1200, total_assets_prev=1000,
    depreciation=50, depreciation_prev=50, sga=220, sga_prev=200, ltd=260, ltd_prev=200,
    current_liabilities=200, current_liabilities_prev=150, net_income=100, cfo=40,
)
EXPECTED = {
    "DSRI": 1.25,                 # (150/1200) / (100/1000)
    "GMI": 0.40 / 0.35,           # (400/1000) / (420/1200)
    "AQI": (1 - 830 / 1200) / 0.30,
    "SGI": 1.2,
    "DEPI": (50 / 350) / (50 / 380),
    "SGAI": (220 / 1200) / 0.20,
    "LVGI": (460 / 1200) / 0.35,
    "TATA": 0.05,                 # (100 - 40) / 1200
}
EXPECTED_M = -1.757952  # summed by hand from the eight terms above

NO_WINSOR = Config(winsor_by="none")


def panel(*overrides):
    return pd.DataFrame([{**BASE, **o} for o in (overrides or [{}])])


def test_hand_computed_firm_year():
    scores, drops = score_panel(panel(), NO_WINSOR)
    assert len(scores) == 1 and drops.empty
    row = scores.iloc[0]
    for idx, want in EXPECTED.items():
        assert row[idx] == pytest.approx(want, abs=1e-9), idx
    assert row["m_score"] == pytest.approx(EXPECTED_M, abs=1e-5)
    assert bool(row["flag"]) is True           # -1.758 > -1.78
    assert row["top_driver"] == "TATA"              # 4.679*0.05 = 0.234 edges out DSRI 0.920*0.25 = 0.230


def test_threshold_is_configurable():
    scores, _ = score_panel(panel(), Config(winsor_by="none", threshold=-1.70))
    assert bool(scores.iloc[0]["flag"]) is False
    assert flagged_table(scores).empty


def test_contributions_sum_to_score():
    scores, _ = score_panel(panel(), NO_WINSOR)
    row = scores.iloc[0]
    assert M_NEUTRAL + sum(row["contrib_" + i] for i in INDICES) == pytest.approx(row["m_score"])
    assert M_NEUTRAL == pytest.approx(-2.48)


@pytest.mark.parametrize("override,fragment", [
    ({"receivables_prev": 0}, "DSRI: zero denominator (receivables_prev = 0)"),
    ({"sales_prev": 0}, "nonpositive sales_prev"),
    ({"cogs": 1200}, "GMI: zero denominator (gross margin = 0)"),
    ({"cogs": 1300}, "GMI: negative gross margin"),
    ({"current_assets_prev": 700}, "AQI: zero denominator"),
    ({"depreciation": 0}, "DEPI: zero denominator (depreciation <= 0)"),
    ({"sga_prev": 0}, "SGAI: zero denominator (sga_prev <= 0)"),
    ({"ltd_prev": 0, "current_liabilities_prev": 0}, "LVGI: zero denominator"),
    ({"total_assets": 0}, "TATA: nonpositive total_assets"),
    ({"cfo": np.nan}, "TATA: missing cfo"),
    ({"sga": np.nan}, "SGAI: missing sga"),
])
def test_bad_inputs_are_dropped_with_reason(override, fragment):
    scores, drops = score_panel(panel(override), NO_WINSOR)
    assert scores.empty
    assert len(drops) == 1
    assert drops.iloc[0]["stage"] == "index_computation"
    assert fragment in drops.iloc[0]["reason"]


def test_all_failing_indices_are_listed():
    _, drops = score_panel(panel({"sales_prev": 0}), NO_WINSOR)
    reason = drops.iloc[0]["reason"]
    for idx in ("DSRI", "GMI", "SGI", "SGAI"):
        assert idx + ":" in reason


def test_no_infinite_scores_ever_survive():
    bad = panel({"receivables_prev": 0}, {"sga_prev": 0}, {"depreciation": 0}, {})
    scores, drops = score_panel(bad, NO_WINSOR)
    assert len(scores) == 1 and len(drops) == 3
    assert np.isfinite(scores[INDICES + ["m_score"]].to_numpy()).all()


def test_neutral_fill_keeps_firm_year_and_records_it():
    scores, drops = score_panel(panel({"sga": np.nan}), Config(winsor_by="none", neutral_fill=("SGAI",)))
    assert drops.empty and len(scores) == 1
    row = scores.iloc[0]
    assert row["SGAI"] == 1.0
    assert "SGAI=1 (missing sga)" in row["imputed"]
    assert row["m_score"] == pytest.approx(EXPECTED_M + COEFS["SGAI"] * (1 - EXPECTED["SGAI"]), abs=1e-5)


def test_neutral_fill_rejects_core_indices():
    with pytest.raises(ValueError):
        Config(neutral_fill=("DSRI",))


@pytest.mark.parametrize("sic,stage_reason", [
    ("6021", "financial"), ("6798", "financial"), ("4911", "utility"), (None, "missing SIC"),
])
def test_financials_and_utilities_excluded(sic, stage_reason):
    scores, drops = score_panel(panel({"sic": sic}), NO_WINSOR)
    assert scores.empty
    assert drops.iloc[0]["stage"] == "industry_exclusion"
    assert stage_reason in drops.iloc[0]["reason"]


def test_sic_boundaries_stay_in():
    scores, _ = score_panel(panel({"sic": "4899"}, {"sic": "5000", "firm_id": "2"}, {"sic": "7000", "firm_id": "3"}), NO_WINSOR)
    assert len(scores) == 3


def test_winsorization_clips_tails_within_year():
    rows = [{"firm_id": str(i), "sales": 1000 + 10 * i, "fiscal_year": 2022} for i in range(101)]
    rows[100]["sales"] = 50_000  # extreme sales growth
    rows.append({"firm_id": "other-year", "sales": 90_000, "fiscal_year": 2023})
    scores, _ = score_panel(panel(*rows), Config(winsor=(0.01, 0.99), winsor_by="year"))
    y22 = scores[scores["fiscal_year"] == 2022]
    raw_p99 = y22["SGI_raw"].quantile(0.99)
    assert y22["SGI_raw"].max() == pytest.approx(50.0)
    assert y22["SGI"].max() == pytest.approx(raw_p99)
    assert y22["SGI"].min() == pytest.approx(y22["SGI_raw"].quantile(0.01))
    # The single 2023 firm is its own group, so it is untouched by 2022's limits.
    y23 = scores[scores["fiscal_year"] == 2023].iloc[0]
    assert y23["SGI"] == pytest.approx(90.0)
    # Scores use winsorized values; the unwinsorized score is kept alongside.
    top = y22.sort_values("SGI_raw").iloc[-1]
    assert top["m_score"] < top["m_score_unwinsorized"]


def test_flagged_table_is_ranked_descending():
    rows = [{"firm_id": str(i), "net_income": 100 + 40 * i} for i in range(5)]
    scores, _ = score_panel(panel(*rows), NO_WINSOR)
    flagged = flagged_table(scores)
    assert list(flagged["rank"]) == list(range(1, len(flagged) + 1))
    assert flagged["m_score"].is_monotonic_decreasing
    assert (flagged["m_score"] > -1.78).all()


# ------------------------------- validation -------------------------------- #
def _scores_for_validation():
    m = [-1.0, -1.5, -2.0, -3.0, -1.2, -2.5, -2.6, -2.7, -2.8, -2.9]
    return pd.DataFrame({
        "firm_id": [str(i).zfill(10) for i in range(1, 11)],
        "ticker": [f"T{i}" for i in range(1, 11)],
        "fiscal_year": 2020, "m_score": m, "top_driver": "SGI",
    })


def test_backtest_rates(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    # 4 known cases scored (M = -1.0, -1.5, -2.0, -3.0), one dropped, one outside the universe.
    labels_csv.write_text("cik,fiscal_year\n1,2020\n2,2020\n3,2020\n4,2020\n77,2020\n88,2019\n")
    drops = pd.DataFrame([{"firm_id": "0000000077", "ticker": "T77", "name": None, "fiscal_year": 2020,
                           "stage": "index_computation", "reason": "SGAI: missing sga"}])
    res = validate.backtest(_scores_for_validation(), validate.load_labels(labels_csv), drops, [-1.78, -2.22])
    assert res["labels_total"] == 6 and res["labels_scored"] == 4
    t = res["table"].set_index("threshold")
    assert t.loc[-1.78, "hits"] == 2 and t.loc[-1.78, "hit_rate"] == pytest.approx(0.5)
    assert t.loc[-1.78, "false_positives"] == 1 and t.loc[-1.78, "false_positive_rate"] == pytest.approx(1 / 6)
    assert t.loc[-2.22, "hits"] == 3 and t.loc[-2.22, "hit_rate"] == pytest.approx(0.75)
    assert t.loc[-2.22, "false_positive_rate"] == pytest.approx(1 / 6)
    status = res["labels"].set_index("firm_id")["status"]
    assert status["77"].startswith("dropped: SGAI: missing sga")
    assert status["88"].startswith("not in scored output")
    report = validate.render_report(res)
    assert "50.0%" in report and "Known cases not scored" in report


def test_backtest_reports_nothing_when_no_label_matches(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("ticker,fiscal_year\nZZZ,2020\n")
    res = validate.backtest(_scores_for_validation(), validate.load_labels(labels_csv))
    assert res["labels_scored"] == 0
    assert math.isnan(res["table"].iloc[0]["hit_rate"])
    assert "no hit rate can be reported" in validate.render_report(res)


def test_wilson_interval_brackets_point_estimate():
    lo, hi = validate.wilson_interval(2, 4)
    assert lo < 0.5 < hi and lo == pytest.approx(0.150, abs=1e-3) and hi == pytest.approx(0.850, abs=1e-3)


# --------------------------------- EDGAR ----------------------------------- #
def _fact(val, end, start=None, accn="A-2022", form="10-K", filed="2023-02-15"):
    return {"start": start, "end": end, "val": val, "accn": accn, "form": form, "filed": filed}


def _synthetic_facts():
    cur, prev = "2022-12-31", "2021-12-31"

    def both(tag, v_cur, v_prev, duration=False):
        s_cur, s_prev = ("2022-01-01", "2021-01-01") if duration else (None, None)
        return {tag: [_fact(v_cur, cur, s_cur), _fact(v_prev, prev, s_prev)]}

    facts = {}
    facts.update(both("Revenues", 1200, 1000, True))
    facts.update(both("GrossProfit", 420, 400, True))
    facts.update(both("AccountsReceivableNetCurrent", 150, 100))
    facts.update(both("AssetsCurrent", 500, 400))
    facts.update(both("PropertyPlantAndEquipmentNet", 330, 300))
    facts.update(both("Assets", 1200, 1000))
    facts.update(both("DepreciationDepletionAndAmortization", 50, 50, True))
    facts.update(both("GeneralAndAdministrativeExpense", 120, 110, True))
    facts.update(both("SellingAndMarketingExpense", 100, 90, True))
    facts.update(both("LongTermDebtNoncurrent", 260, 200))
    facts.update(both("LiabilitiesCurrent", 200, 150))
    facts.update(both("NetIncomeLoss", 100, 80, True))
    facts.update(both("NetCashProvidedByUsedInOperatingActivities", 40, 70, True))
    # Noise that must be ignored: a quarterly duration, a later amendment with
    # restated numbers, and the next year's 10-K restating 2022 as a comparative.
    facts["Revenues"].append(_fact(333, cur, "2022-10-01"))
    facts["Revenues"].append(_fact(9999, cur, "2022-01-01", accn="A-2022-AMEND", form="10-K/A", filed="2023-06-01"))
    facts["Assets"].append(_fact(8888, cur, accn="A-2022-AMEND", form="10-K/A", filed="2023-06-01"))
    facts["Assets"] += [_fact(1500, "2023-12-31", accn="A-2023", filed="2024-02-15"),
                        _fact(7777, cur, accn="A-2023", filed="2024-02-15")]
    return facts


def test_edgar_rows_come_from_the_original_10k():
    ident = {"firm_id": "0000000001", "ticker": "AAA", "name": "Alpha Co", "sic": "3570"}
    rows, drops = edgar.firm_year_rows(ident, _synthetic_facts(), 2021, 2023)
    by_fy = {r["fiscal_year"]: r for r in rows}
    row = by_fy[2022]
    assert row["sales"] == 1200 and row["sales_prev"] == 1000          # not the 10-K/A or quarterly value
    assert row["total_assets"] == 1200 and row["total_assets_prev"] == 1000
    assert row["cogs"] == 780 and row["cogs_prev"] == 600              # sales - GrossProfit
    assert row["sga"] == 220 and row["sga_prev"] == 200                # G&A + selling
    assert "sga: built from components" in row["notes"]
    assert row["source_ref"].startswith("10-K A-2022")
    # FY2021 has no filing in the synthetic data; FY2023's filing lacks most items.
    reasons = {d["fiscal_year"]: d["reason"] for d in drops}
    assert "no 10-K with a tagged balance sheet" in reasons[2021]
    scores, model_drops = score_panel(pd.DataFrame(rows), NO_WINSOR)
    assert scores.iloc[0]["m_score"] == pytest.approx(EXPECTED_M, abs=1e-5)
    assert set(model_drops["fiscal_year"]) == {2023}


def test_edgar_assumes_zero_debt_only_with_a_note():
    facts = _synthetic_facts()
    del facts["LongTermDebtNoncurrent"]
    ident = {"firm_id": "0000000001", "ticker": "AAA", "name": "Alpha Co", "sic": "3570"}
    rows, _ = edgar.firm_year_rows(ident, facts, 2022, 2022)
    assert rows[0]["ltd"] == 0 and rows[0]["ltd_prev"] == 0
    assert "ltd: no long-term debt tag in filing, assumed 0" in rows[0]["notes"]


def test_fiscal_year_label_follows_compustat_convention():
    assert edgar.fiscal_year_label("2023-09-30") == 2023
    assert edgar.fiscal_year_label("2023-01-28") == 2022
    assert edgar.fiscal_year_label("2023-05-31") == 2022
    assert edgar.fiscal_year_label("2023-06-30") == 2023


# ------------------------------- Compustat --------------------------------- #
def test_compustat_pairs_consecutive_years_and_logs_gaps(tmp_path):
    cols = "gvkey,tic,conm,fyear,datadate,sich,sale,cogs,rect,act,ppent,at,dp,xsga,dltt,lct,ib,oancf,indfmt,datafmt,consol,popsrc"
    lines = [
        cols,
        "001000,AAA,ALPHA CO,2021,20211231,3570,1000,600,100,400,300,1000,50,200,200,150,80,70,INDL,STD,C,D",
        "001000,AAA,ALPHA CO,2022,20221231,3570,1200,780,150,500,330,1200,50,220,260,200,100,40,INDL,STD,C,D",
        "001000,AAA,ALPHA CO,2022,20221231,3570,1,1,1,1,1,1,1,1,1,1,1,1,FS,STD,C,D",  # filtered out
        "002000,BBB,BETA CO,2020,20201231,2000,900,500,90,300,200,800,40,150,100,120,60,50,INDL,STD,C,D",
        "002000,BBB,BETA CO,2022,20221231,2000,950,520,95,310,210,820,41,155,105,125,62,55,INDL,STD,C,D",
    ]
    path = tmp_path / "funda.csv"
    path.write_text("\n".join(lines) + "\n")
    pnl, drops = compustat.build_panel(path, 2022, 2022)
    assert list(pnl["firm_id"]) == ["001000"]
    assert list(drops["firm_id"]) == ["002000"] and "no consecutive prior fiscal year" in drops.iloc[0]["reason"]
    scores, _ = score_panel(pnl, NO_WINSOR)
    assert scores.iloc[0]["m_score"] == pytest.approx(EXPECTED_M, abs=1e-5)


# -------------------------------- drill-down ------------------------------- #
def test_drilldown_renders_one_page():
    rows = []
    for fy in (2020, 2021, 2022):
        for i in range(6):
            rows.append({"firm_id": str(i), "ticker": f"T{i}", "name": f"Firm {i} & Co", "fiscal_year": fy,
                         "sales": 1200 + 30 * i + 10 * (fy - 2020), "net_income": 100 + 15 * i})
    scores, _ = score_panel(panel(*rows), NO_WINSOR)
    page = drilldown.render(scores, "t3")
    assert page.startswith("<!DOCTYPE html>")
    assert "Firm 3 &amp; Co (T3)" in page and "fiscal year 2022" in page
    assert page.count("<svg") == 9                      # M-Score + 8 indices
    assert "6 firms in SIC 35xx" in page
    for idx in INDICES:
        assert f"<td>{idx}</td>" in page
    with pytest.raises(ValueError):
        drilldown.render(scores, "NOPE")


def test_edgar_sales_uses_largest_revenue_tag():
    # Seen in real filings: `Revenues` tagged on a side item (or as 0) beside the real total.
    facts = _synthetic_facts()
    facts["RevenueFromContractWithCustomerExcludingAssessedTax"] = facts.pop("Revenues")
    facts["Revenues"] = [_fact(130, "2022-12-31", "2022-01-01"), _fact(110, "2021-12-31", "2021-01-01")]
    ident = {"firm_id": "0000000001", "ticker": "AAA", "name": "Alpha Co", "sic": "3570"}
    rows, _ = edgar.firm_year_rows(ident, facts, 2022, 2022)
    assert rows[0]["sales"] == 1200 and rows[0]["sales_prev"] == 1000
    assert "sales=RevenueFromContractWithCustomerExcludingAssessedTax" in rows[0]["tags_used"]
