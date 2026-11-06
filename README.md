# Beneish M-Score screen

Computes the Beneish (1999) eight-variable M-Score for a universe of companies, flags
firm-years above a threshold, logs every firm-year it could not score and why, and
produces a one-page drill-down per ticker.

**Status.** The pipeline runs end to end on live SEC EDGAR data and the arithmetic is
unit-tested against a hand-worked example. **No back-test has been run**: the tool ships
no list of known manipulators, so there is no hit rate or false positive rate to report
until you supply one (see [Validation](#validation)).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC requires this for automated access

# 1. Score a universe (tickers or CIKs, one per line)
.venv/bin/python -m beneish run --source edgar --universe my_universe.txt --start-year 2015 --end-year 2024

# 2. One-page drill-down for a ticker (opens in any browser, prints on one sheet)
.venv/bin/python -m beneish drilldown SMCI

# 3. Back-test against your known cases
.venv/bin/python -m beneish validate --labels my_labels.csv

# Tests
.venv/bin/python -m pytest -q
```

Compustat instead of EDGAR: export `comp.funda` to CSV and run
`python -m beneish run --source compustat --input funda.csv --start-year 2015 --end-year 2024`.

## Choices made where the brief was left open

| Brief item | What the tool does | How to change it |
|---|---|---|
| Data source | SEC EDGAR XBRL company-facts API (free, no licence). A Compustat CSV adapter is included but has only been tested on synthetic rows, not a real WRDS export. | `--source compustat --input FILE` |
| Universe | Whatever list you pass. Russell 3000 membership is licensed and is not bundled. `examples/sample_universe.txt` is about 100 large caps used for the smoke test. | `--universe FILE` |
| Fiscal years | No default; you must pass a range. XBRL coverage on EDGAR starts around fiscal 2009-2011. | `--start-year`, `--end-year` |
| Threshold | `M > -1.78`. The validation report always shows both -1.78 and -2.22. | `--threshold -2.22` |
| Winsorization | Each index clipped at the 1st/99th percentile **within fiscal year** (no look-ahead across years). | `--winsor 0.01 0.99`, `--winsor-by year\|pooled\|none` |
| Known cases | None bundled. | `validate --labels FILE` |

On the threshold: the model is a probit, so `M = -1.78` corresponds to an estimated
manipulation probability of about 3.8% and `-2.22` to about 1.3%. Both cutoffs are quoted
for this model; `-2.22` flags more firms. Use whichever you used before.

## Formula

```
M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
          + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
```

Subscript `t` is the fiscal year being scored, `t-1` the year before.

| Index | Definition |
|---|---|
| DSRI | (Receivables_t / Sales_t) / (Receivables_t-1 / Sales_t-1) |
| GMI | GrossMargin_t-1 / GrossMargin_t, where GrossMargin = (Sales - COGS) / Sales |
| AQI | [1 - (CurrentAssets_t + PPE_t) / TotalAssets_t] / [1 - (CurrentAssets_t-1 + PPE_t-1) / TotalAssets_t-1] |
| SGI | Sales_t / Sales_t-1 |
| DEPI | [Dep_t-1 / (Dep_t-1 + PPE_t-1)] / [Dep_t / (Dep_t + PPE_t)] |
| SGAI | (SGA_t / Sales_t) / (SGA_t-1 / Sales_t-1) |
| LVGI | [(LTD_t + CurrentLiabilities_t) / TotalAssets_t] / [(LTD_t-1 + CurrentLiabilities_t-1) / TotalAssets_t-1] |
| TATA | (Income from continuing operations_t - Cash flow from operations_t) / TotalAssets_t |

TATA uses the cash-flow definition of accruals. Beneish's original paper built accruals
from balance-sheet changes (change in working capital less depreciation); the cash-flow
version is the one normally used with post-1988 data and needs fewer line items.

A firm whose ratios did not change and whose accruals are zero scores **-2.48** (all ratio
indices equal 1, TATA equal 0). The `contrib_*` columns give each index's distance from
that baseline, `coefficient x (index - neutral)`, so `m_score = -2.48 + sum(contrib_*)`.
`top_driver` names the largest one.

## Data mapping

| Line item | Feeds | EDGAR us-gaap tags, in order tried | Compustat |
|---|---|---|---|
| Sales | DSRI, GMI, SGI, SGAI | The largest of `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `...IncludingAssessedTax`, `SalesRevenueNet`, `SalesRevenueGoodsNet` | `sale` |
| COGS | GMI | Sales - `GrossProfit` if tagged; else `CostOfRevenue`, `CostOfGoodsAndServicesSold`, `CostOfGoodsSold` | `cogs` |
| Receivables | DSRI | `AccountsReceivableNetCurrent`, `ReceivablesNetCurrent`, `AccountsNotesAndLoansReceivableNetCurrent`, `AccountsAndOtherReceivablesNetCurrent` | `rect` |
| Current assets | AQI | `AssetsCurrent` | `act` |
| Net PP&E | AQI, DEPI | `PropertyPlantAndEquipmentNet`, `PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization` | `ppent` |
| Total assets | AQI, LVGI, TATA | `Assets` | `at` |
| Depreciation | DEPI | `Depreciation`, `DepreciationDepletionAndAmortization`, `DepreciationAndAmortization`, `DepreciationAmortizationAndAccretionNet` | `dp` (less `am` if present) |
| SG&A | SGAI | `SellingGeneralAndAdministrativeExpense`; else `GeneralAndAdministrativeExpense` + (`SellingAndMarketingExpense` or `SellingExpense`) | `xsga` |
| Long-term debt | LVGI | `LongTermDebtNoncurrent`, `LongTermDebtAndCapitalLeaseObligations`, `LongTermNotesPayable`, `SeniorLongTermNotes`, `ConvertibleNotesPayableNoncurrent`, `ConvertibleLongTermNotesPayable`, `ConvertibleDebtNoncurrent`, `LongTermLineOfCredit`, `OtherLongTermDebtNoncurrent`; else `LongTermDebt` - `LongTermDebtCurrent`; else `DebtLongtermAndShorttermCombinedAmount` - `DebtCurrent`; else 0 with a note | `dltt` |
| Current liabilities | LVGI | `LiabilitiesCurrent` | `lct` |
| Income | TATA | `IncomeLossFromContinuingOperations`, `ProfitLoss`, `NetIncomeLoss`, `NetIncomeLossAvailableToCommonStockholdersBasic` | `ib` (else `ibc`) |
| Operating cash flow | TATA | `NetCashProvidedByUsedInOperatingActivities`, `...ContinuingOperations` | `oancf` |

How EDGAR rows are built:

- **Both years come from the same original 10-K.** Year `t` and the `t-1` comparative are
  read from one filing, so they share an accounting basis and the same tags. The original
  10-K is used, not a later 10-K/A, because a restatement removes the thing the model is
  looking for. A 10-K/A is used only when no original 10-K has XBRL data.
- Only full-year facts (350-380 days) are used. Transition-period reports (10-KT) are ignored.
- Fiscal year follows the Compustat convention: a year ending January-May is labelled
  with the prior calendar year.
- A tag reported for both years is preferred over one reported for only one.
- `tags_used`, `notes` and `source_ref` in `scores.csv` record, per firm-year, exactly
  which tag fed each item, any fallback taken, and the accession number.
- Industry comes from the SIC code on the SEC submissions record. SIC 6000-6999
  (financials) and 4900-4999 (utilities) are excluded before any download.

## Missing data and zero denominators

Nothing is silently imputed or silently dropped. A firm-year is scored only if all eight
indices can be computed. Otherwise it goes to `dropped.csv` with every failing index named:

```
firm_id,ticker,name,fiscal_year,stage,reason
0001341439,ORCL,ORACLE CORP,2024,index_computation,GMI: missing cogs
```

`stage` is `universe` (ticker not found), `industry_exclusion`, `data` (no usable filing or
no prior year), or `index_computation`. A blank `fiscal_year` means the whole firm was dropped.

Rules applied in `beneish/model.py`:

| Condition | Treatment |
|---|---|
| Any required input missing | Drop: `<INDEX>: missing <item>` |
| Sales or total assets <= 0 in either year | Drop |
| Receivables_t-1 = 0, SGA_t-1 <= 0, Depreciation_t <= 0, prior-year leverage = 0, prior-year asset-quality ratio = 0 | Drop: `zero denominator (...)` |
| Gross margin = 0 in year t, or negative in either year | Drop: the ratio of two margins is meaningless when one is negative |
| Current assets + PP&E exceed total assets | Drop (inconsistent inputs) |
| No long-term debt tag in an EDGAR filing | Treated as 0, recorded in `notes` (filers without debt do not report the tag) |

Optional: `--neutral-fill GMI AQI DEPI SGAI` sets the listed indices to 1 instead of
dropping the firm-year, and records each instance in the `imputed` column. Off by default.
Beneish's paper describes a similar treatment for AQI, DEPI and SGAI; I am working from
memory of that footnote, so check it against the paper before citing it. Allowing GMI is
my extension for XBRL data, where cost of sales is the most common gap.

## Outputs

All in `output/` (override with `--out`):

| File | Contents |
|---|---|
| `flagged.csv` | Firm-years with `M > threshold`, ranked highest first, with all eight indices, their contributions and `top_driver` |
| `scores.csv` | Every scored firm-year, same columns. `DSRI`...`TATA` are winsorized (used in the score); `*_raw` are before winsorization; `m_score_unwinsorized` is kept for comparison |
| `dropped.csv` | The drop log |
| `fundamentals.csv` | The line items behind each firm-year (`sales`, `sales_prev`, ...), for auditing |
| `drilldown_<TICKER>.html` | One page: M-Score and each index against the industry median over three years, plus a table for the latest year |
| `validation.md`, `validation.csv` | Back-test report (only after you run `validate`) |

Industry median in the drill-down is the median across scored firms with the same
two-digit SIC code in the same fiscal year. With fewer than 5 such peers (`--min-peers`)
it falls back to all scored firms, and the page says which was used.

## Validation

Not run. To run it, create a CSV of known manipulation firm-years (template:
`examples/labels_template.csv`) with `fiscal_year` and `cik` (or `gvkey`) and/or `ticker`:

```bash
python -m beneish validate --labels my_labels.csv --thresholds -1.78 -2.22
```

The report gives, per threshold:

- **Hit rate**: known cases with M above the threshold / known cases scored, with a 95% Wilson interval.
- **False positive rate**: unlabelled firm-years above the threshold / unlabelled firm-years scored.
- Precision, the M-Score of every known case, and every known case that was *not*
  scored with the reason from the drop log. Unscored cases are excluded from the hit rate
  and listed, not hidden.

Reading the numbers:

- Unlabelled is not the same as clean, so the false positive rate is an upper bound.
- Label the fiscal years in which the misstatement occurred, not the year of the
  enforcement action.
- SEC AAERs are published on sec.gov; the standard research dataset built from them
  (Dechow, Ge, Larson and Sloan) is distributed by its authors on request.
- Most well-known cases predate XBRL and cannot be scored from EDGAR. Use Compustat for
  those.

## What was actually checked

- 34 unit tests pass: a firm-year worked by hand (M = -1.758), every zero-denominator
  and missing-input rule, the SIC exclusions, winsorization, the validation arithmetic
  on synthetic labels, EDGAR filing selection, the Compustat pairing, and the drill-down.
- A live EDGAR run on `examples/sample_universe.txt` (106 CIKs, fiscal 2015-2024):
  7 firms excluded as financials or utilities, 632 firm-years scored across 75 firms,
  22 flagged at -1.78, 358 firm-years dropped. Of the drops, 197 lacked a cost-of-sales
  tag and 133 lacked an SG&A tag (some lacked both). Median M-Score was -2.62 and the median of each ratio
  index was within 0.06 of 1. The files in `output/` are from this run.
- Apple's fiscal 2023 inputs in `fundamentals.csv` were compared with its 10-K and match.

That run is a pipeline check on a hand-picked list of large caps. It is not a result
about your universe and it says nothing about the model's accuracy.

## Limitations

- **A flag is a screen, not a finding.** High growth, acquisitions, and business-model
  shifts all push the score up for legitimate reasons. Use the drill-down to see which
  index is responsible before drawing conclusions.
- **The coefficients are from 1982-1992 data.** They are not re-estimated here.
- **EDGAR coverage is materially lower than Compustat's.** XBRL is as-reported: a firm
  that does not present a cost-of-sales or SG&A line (airlines, oil majors, restaurants,
  many service firms) has no tag for it, and the firm-year is dropped. About a third of
  firm-years in the smoke test were lost this way. Compustat standardizes those items.
  The dropped firms are not a random sample, so screening statistics on EDGAR data are
  tilted toward firms with conventional income statements.
- **Definitions differ from Compustat.** XBRL cost of sales usually includes depreciation;
  the receivables tag is trade receivables where available; depreciation may be D&A when
  no pure depreciation tag exists. Scores will not match a Compustat-based calculation exactly.
- **Long-term debt is the weakest mapping.** Filers tag debt many ways. When no tag
  matches, debt is taken as 0 and noted (34 of 632 scored firm-years in the smoke test,
  some genuinely debt-free). LVGI carries the smallest coefficient after DEPI and SGAI, so
  the effect on M is limited, but check `notes` for any firm you care about.
- **SIC codes are current, not historical**, and are the SEC's assignment, which can
  differ from Compustat's.
- **Acquisitions and tag changes.** Same-filing comparatives keep the two years on one
  basis, but an acquisition still moves SGI, AQI and DSRI mechanically.
- **Foreign filers.** 20-F and 40-F filers and IFRS reporters are not covered by the EDGAR loader.
- **Small universes.** Percentile winsorization does little with fewer than about 100
  firms per year, and industry medians fall back to the whole universe.
- **Survivorship.** A universe built from today's ticker list omits delisted firms,
  which is where many manipulators end up. Pass CIKs of delisted firms to include them.
- **SEC access.** www.sec.gov rejects requests without a descriptive User-Agent, and
  ticker lookup needs it. data.sec.gov is rate-limited; the client stays under 7 requests
  a second and caches trimmed responses in `cache/edgar/` (`--refresh` to re-download).

## Layout

```
beneish/model.py       indices, drop rules, winsorization, scoring
beneish/edgar.py       EDGAR client, tag mapping, filing selection
beneish/compustat.py   Compustat CSV adapter
beneish/validate.py    back-test against a labels file
beneish/drilldown.py   one-page HTML report
beneish/__main__.py    command line
tests/test_beneish.py  unit tests (synthetic data only)
examples/              sample universe, labels template
```

## Reference

Beneish, M. D. (1999). The Detection of Earnings Manipulation. *Financial Analysts Journal*, 55(5), 24-36.
