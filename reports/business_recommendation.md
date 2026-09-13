# ReturnFlowAI — Return Forecasting & Business Impact Report

**Author:** Yashvardhan Shah
**Dataset:** Olist Brazilian E-Commerce Public Dataset
**Code:** [GitHub Repository](https://github.com/yashvardhanshah/ReturnFlow) · [Live Dashboard](https://returnflow.streamlit.app/)

---

## Executive Summary

Every return costs money twice over — once in the original fulfillment, again in reverse shipping, restocking, and lost or discounted resale. This project set out to answer three questions a business owner actually cares about: **how many returns are coming, why are they happening, and what's the fastest lever to reduce them?**

Using 85 weeks of order data (99,441 orders), an XGBoost forecasting model predicts weekly return volume within ~21% average error, enough accuracy to plan reverse-logistics staffing and budget instead of reacting to surprises. Statistical testing confirmed two concrete, actionable drivers: **delivery buffer** and **product category**. A retrospective comparison found that orders shipped with tighter delivery buffer return at **nearly double the rate** of orders shipped with more buffer (17.3% vs. 8.9%) — the single largest lever identified in this analysis, worth an estimated **~43 fewer returns per week** if closed. SQL-based seller analysis went one step further, naming specific sellers whose return rates run 2–5x their category average — giving operations a short, immediately actionable list rather than a vague category-wide policy.

**In short: this analysis doesn't just describe the return problem — it points to where the money is being lost and which fixes would recover the most of it fastest.**

**Scope note:** this dataset has no explicit "returned" flag. All return-related findings use a **return-proxy** (orders with a 1–2 star review or a cancelled status) as a stand-in for genuine returns — a standard, reasonable approximation, but not ground truth. All figures should be read as directional, decision-grade estimates rather than final audited numbers.

---

## 1. Forecasting: SARIMA vs. XGBoost

**Why this matters for the business:** every week you can't predict return volume, you're either overstaffing reverse-logistics (wasted labor cost) or understaffing it (backlogs, unhappy customers, slower refunds). A reliable weekly forecast turns that into a planned, budgeted cost instead of a surprise.

**Method:** weekly order/return-proxy counts were aggregated from order-level data, gap-filled to a continuous calendar, and trimmed of unreliable low-volume weeks (platform ramp-up and data-collection tail-off), leaving 85 usable weeks. An Augmented Dickey-Fuller test confirmed the series was non-stationary (p = 0.057 with a trend term) and became stationary after first-order differencing (p < 0.00001), which informed the SARIMA specification used below. The final 12 weeks (Jun–Aug 2018) were held out as a test set; both models were trained only on the preceding 73 weeks and evaluated on identical, unseen test weeks.

- **SARIMA:** `auto_arima` initially selected a model with no meaningful autoregressive structure, and one-shot 12-week-ahead forecasting produced unusable results (83.9% MAPE), including implausible negative predictions. Diagnosing this: the model lacked a trend term and had no way to represent the known Black Friday demand spike. Adding an exogenous spike-week flag, a trend component, and switching to **walk-forward (rolling) forecasting** — predicting one week at a time and updating with the true observed value before forecasting the next — improved performance to **29.11% MAPE, 60.27 RMSE**. A further attempt to add a lag-1 feature made results worse (35.08% MAPE) and was reverted; SARIMA's own AR/MA structure already captured this information, and the addition was redundant.

- **XGBoost:** engineered features included 1–3 week lags, a 4-week rolling mean, calendar features (month, week-of-year), and the same spike-week flag. Trained once on 69 weeks (4 fewer than SARIMA due to lag-feature warm-up), evaluated on the same 12 test weeks. Achieved **20.93% MAPE, 49.57 RMSE** — the stronger of the two models.

**Result:**

| Model | MAPE | RMSE |
|---|---|---|
| SARIMA (walk-forward, spike flag, trend) | 29.11% | 60.27 |
| **XGBoost (lag + calendar features)** | **20.93%** | **49.57** |

**Recommendation:** deploy XGBoost as the primary weekly return-volume forecast for reverse-logistics staffing and budget planning, with SARIMA retained as a secondary/sanity-check model.

---

## 2. Root-Cause Analysis: Hypothesis Testing

**Why this matters for the business:** knowing returns are high is not actionable. Knowing *what specifically* drives them tells you where to spend money to fix it.

### 2.1 Delivery buffer

Orders were split by return-proxy status and compared on delivery buffer (days delivered ahead of the estimated date; more negative = more buffer). A Welch's t-test found:

- Non-returned orders: **-12.86 days** average buffer
- Returned orders: **-5.13 days** average buffer
- **t = 54.17, p < 0.000001**

Returned orders arrived, on average, with **~7.7 fewer days of buffer** than non-returned orders. To rule out product category as a confound, the comparison was repeated within each of the 26 major categories individually — the direction of the effect held in **every single category**, meaning this is not an artifact of category mix.

**Caveat:** this is an association from observational data, not proof of causation. Tighter delivery windows likely correlate with other factors (order timing, seller/logistics strain during high-volume periods) that could also independently affect returns.

### 2.2 Product category

A chi-square test of independence between product category and return-proxy status found a statistically significant relationship (**χ² = 299.41, p < 0.000001**, 25 degrees of freedom, categories with 500+ orders).

Highest return-rate categories: `moveis_escritorio` (office furniture, 22.5%), `cama_mesa_banho` (bed/bath linens, 16.4%), `moveis_decoracao` (home decor, 16.2%).
Lowest return-rate categories: `livros_interesse_geral` (general books, 8.3%), `malas_acessorios` (luggage, 8.9%), `papelaria` (stationery, 11.3%).

**Recommendation:** prioritize quality-control review (packaging, handling, seller vetting) for office furniture and home-decor categories, which sit well above the ~14% platform average return rate and are the categories where a fix would move the most money.

---

## 3. Simulated A/B Test: Delivery-Speed Impact — The Biggest Lever

**Why this matters for the business:** this is the single highest-leverage finding in the analysis — it puts an estimated size on the opportunity, not just a direction.

Since a real controlled experiment on historical data isn't possible, orders were split into two comparable groups using the platform-wide median delivery buffer (-12 days) as a natural cutoff, and treated as a retrospective "high buffer" vs. "low buffer" comparison.

| Group | Orders | Return Rate |
|---|---|---|
| High buffer (≤ -12 days) | 52,589 | 8.90% |
| Low buffer (> -12 days) | 43,887 | 17.29% |

Difference statistically significant (χ² = 1515.04, p < 0.000001).

**Simulated impact:** if low-buffer orders had instead matched the high-buffer group's return rate, an estimated **3,681 fewer returns** would have occurred across the dataset — a **48.5% relative reduction** in returns for that group, or roughly **43 fewer returns per week** if applied platform-wide.

**Turning this into a dollar figure:** this report does not have access to Olist's actual per-return cost (reverse shipping, restocking, discounted resale, lost margin), so no dollar amount is invented here. The figure below is a ready-to-use formula — plug in the real cost per return and the savings estimate updates instantly:

> **Estimated annual savings = 43 returns/week × 52 weeks × (cost per return)**
> *Example: at a conservative $8 cost per return, that's ~$17,900/year. At $20 per return, ~$44,700/year — from delivery-reliability improvements alone.*

**Caveat:** this is a simulated, correlational estimate based on historical groups, not a randomized controlled experiment. The true causal effect of a real delivery-speed intervention could be smaller (or larger) than this estimate, as other factors correlated with delivery buffer (seller reliability, regional logistics quality, order complexity) may also contribute to the gap.

**Recommendation:** delivery-reliability investment (buffer/SLA improvements) is the highest-leverage lever identified in this analysis and is worth piloting as a real, measurable experiment on a subset of sellers or regions to convert this estimate into a confirmed number.

---

## 4. SQL Analytics: Seller & Category Return Risk

**Why this matters for the business:** fixing "office furniture returns are high" is a slow, expensive, category-wide project. Fixing "this one seller has a 58.8% return rate" is a phone call.

**Objective:** move beyond category-level findings to identify specific underperforming sellers, using SQL window functions (`AVG() OVER`, `RANK() OVER`) against a local SQLite database of 114,092 order-items.

Sellers with fewer than 15 items sold were excluded to avoid unreliable small-sample return rates. Among sellers with sufficient volume, several showed return rates **2–5x their category average**:

- Seller `2eb70248d66e0e3ef83659f71b244378` (`informatica_acessorios`): **58.8%** return rate across 68 items vs. an **18.8%** category average.
- Seller `4342d4b2ba6b161468c63a7e7cfce593` (`relogios_presentes`): 90.0% return rate across 20 items vs. 17.8% category average.

These are large enough samples to be confident this reflects genuine seller-level quality issues, not random noise, and the pattern spans multiple unrelated categories — suggesting seller-specific rather than category-wide problems in these cases.

**Recommendation:** flag sellers with return rates significantly above their category benchmark (using the ranking query in `sql/seller_category_analysis.sql`) for a direct quality review process — the fastest, cheapest fix available from this data, since it targets a small number of accounts rather than an entire category.

---

## Overall Recommendations (Ranked by Speed & Impact)

1. **Review flagged sellers immediately** — smallest effort, fastest payoff. A short list of named accounts, not a policy change.
2. **Pilot a delivery-speed improvement** for a subset of sellers or regions — the largest estimated impact (~43 fewer returns/week), but needs a real pilot to confirm before scaling company-wide.
3. **Adopt XGBoost** for weekly return-volume forecasting — turns reverse-logistics staffing from reactive to planned, reducing both overstaffing waste and understaffing backlogs.
4. **Prioritize office furniture and home-decor categories** for packaging/handling quality review — the categories where a fix moves the most money.
5. **Institutionalize the SQL seller-ranking query** as a recurring monthly or quarterly process, so underperforming sellers are caught early and automatically, not just once.

## Limitations

- Returns are approximated via review score/cancellation, not an explicit return flag — findings should be validated against actual return records if accessible.
- All causal-sounding findings (delivery buffer, A/B simulation) are associations from observational data; true causal effects, and any dollar savings, should be validated via a real controlled pilot before being used in a formal budget.
- The dataset covers Oct 2016–Aug 2018; patterns may have shifted since, and forecasting models would need retraining on current data before production use.

---

## Summary for Business Stakeholders

- **Forecasting accuracy:** the recommended model (XGBoost) predicts weekly return volume within **~21% average error**, a meaningful improvement over the classical statistical baseline (~29% error) — accurate enough to plan reverse-logistics staff and budget instead of reacting after the fact.
- **What drives returns:** orders delivered with less delivery buffer are significantly more likely to be returned, holding true across every product category tested. Category itself also matters — office furniture returns at **22.5%**, more than 2.5x the rate of books (**8.3%**).
- **Where the money is:** orders with strong delivery buffer had a return rate of **8.9%**, versus **17.3%** for tighter-buffer orders — nearly double. Closing that gap could mean roughly **43 fewer returns every week**, a **~48.5% reduction** for the affected orders — translating directly into lower reverse-shipping and restocking costs (see the savings formula in Section 3).
- **The fastest win:** a small number of specific, named sellers were found with return rates **2–5x higher** than their category peers on reliable sample sizes — ready for immediate review, at essentially no analysis cost since the query is already built and can be rerun anytime.
- **Bottom line:** this analysis identifies three real levers to reduce returns — one fast and cheap (seller review), one high-impact but needing validation (delivery-speed pilot), and one operational (better forecasting to cut staffing waste) — together representing the clearest, evidence-backed path from "returns are a cost center" to "returns are a managed, shrinking line item."

**Recommendation:** adopt the XGBoost model for weekly return forecasting, launch a controlled delivery-speed pilot on a subset of sellers/regions to validate the ~48.5% impact estimate and convert it into a confirmed dollar figure, and route the flagged high-risk sellers into an immediate quality review process — starting with the seller review, since it requires no further validation and can begin today.
