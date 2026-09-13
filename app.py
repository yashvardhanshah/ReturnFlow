import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ReturnFlowAI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STATIC DATA — all figures below are real results generated
# in the project notebooks (01-06), not placeholders.
# ============================================================

FORECAST_DF = pd.DataFrame({
    "week": ["2018-06-04", "2018-06-11", "2018-06-18", "2018-06-25", "2018-07-02",
             "2018-07-09", "2018-07-16", "2018-07-23", "2018-07-30", "2018-08-06",
             "2018-08-13", "2018-08-20"],
    "Actual": [182, 159, 133, 147, 105, 145, 186, 169, 286, 251, 163, 90],
    "SARIMA": [88.65, 239.19, 153.63, 111.71, 120.11, 71.69, 164.72, 183.18,
               182.75, 323.00, 229.46, 106.00],
    "XGBoost": [136.17, 193.34, 134.40, 131.70, 154.97, 126.10, 141.18, 187.45,
                148.36, 252.57, 198.16, 114.25],
})
FORECAST_DF["week"] = pd.to_datetime(FORECAST_DF["week"])

SARIMA_MAPE, SARIMA_RMSE = 29.11, 60.27
XGB_MAPE, XGB_RMSE = 20.93, 49.57

CATEGORY_DF = pd.DataFrame({
    "category": ["moveis_escritorio", "cama_mesa_banho", "moveis_decoracao",
                 "informatica_acessorios", "bebes", "fashion_bolsas_e_acessorios",
                 "pet_shop", "papelaria", "malas_acessorios", "livros_interesse_geral"],
    "return_rate": [0.224506, 0.163677, 0.162392, 0.159760, 0.154984,
                     0.115426, 0.113850, 0.112903, 0.088954, 0.082515],
    "group": ["Highest 5"] * 5 + ["Lowest 5"] * 5
})
OVERALL_AVG_RETURN_RATE = 0.1409

# Buffer group stats (A/B simulation baseline)
LOW_BUFFER_ORDERS = 43887
HIGH_BUFFER_ORDERS = 52589
LOW_BUFFER_RATE = 0.172921
HIGH_BUFFER_RATE = 0.089049
TOTAL_WEEKS = 85

SELLER_DF = pd.DataFrame({
    "seller_id": ["4342d4b2ba6b161468c63a7e7cfce593", "b1b3948701c5c72445495bd161b83a4c",
                  "1ca7077d890b907f89be8c954a02686a", "710e3548e02bc1d2831dfc4f1b5b14d4",
                  "2709af9587499e95e803a6498a5a56e9", "bb135baca94c82fcb731335ad5b04a03",
                  "d12c926d74ceff0a90a21184466ce161", "5bc55dbe2f12b6af6d83ed46023e0dc8",
                  "e250d617a0ad591ba9bd663e584a895d", "c6381d2d013342748761e906d45aff76",
                  "7c67e1448b00f6e969d365cea6b010ab", "2eb70248d66e0e3ef83659f71b244378",
                  "ecccfa2bb93b34a3bf033cc5d1dcdc69", "c6a7539d424a8402232c2228d7a03c5e",
                  "973f21788dfab357250f69a8dcb7ddee"],
    "category": ["relogios_presentes", "automotivo", "brinquedos", "informatica_acessorios",
                 "beleza_saude", "moveis_decoracao", "perfumaria", "esporte_lazer",
                 "moveis_decoracao", "utilidades_domesticas", "esporte_lazer",
                 "informatica_acessorios", "beleza_saude", "utilidades_domesticas", "esporte_lazer"],
    "items_sold": [20, 18, 25, 30, 37, 29, 20, 19, 20, 25, 18, 68, 19, 15, 21],
    "seller_return_rate": [0.9000, 0.7778, 0.7600, 0.7667, 0.6216, 0.6552, 0.6000,
                            0.5789, 0.6000, 0.5600, 0.5556, 0.5882, 0.5263, 0.5333, 0.5238],
    "category_avg_return_rate": [0.1780, 0.1559, 0.1428, 0.1880, 0.1332, 0.1898, 0.1601,
                                  0.1527, 0.1898, 0.1512, 0.1527, 0.1880, 0.1332, 0.1512, 0.1527],
})
SELLER_DF["gap_vs_category"] = SELLER_DF["seller_return_rate"] - SELLER_DF["category_avg_return_rate"]

REPORT_TEXT = """# ReturnFlowAI — Return Forecasting & Business Impact Report

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

**Turning this into a dollar figure:** this report does not have access to Olist's actual per-return cost (reverse shipping, restocking, discounted resale, lost margin), so no dollar amount is invented here. The figure below is a ready-to-use formula — plug in the real cost per return and the savings estimate updates instantly (see the interactive A/B Simulator tab in the dashboard):

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
"""

# ============================================================
# THEME / DESIGN SYSTEM
# ============================================================

INK        = "#E7EBF2"   # near-white — headings/body text
SUBTLE     = "#93A0B4"   # muted slate — secondary text
SURFACE    = "#141A26"   # card surfaces
SURFACE_2  = "#1B2333"   # slightly raised surface (tab bar, inputs)
CANVAS     = "#0B0F17"   # page background (near-black navy)
BORDER     = "#262F42"
PRIMARY    = "#3E6FA5"   # steel blue — brand
PRIMARY_D  = "#294B70"
PRIMARY_L  = "#7FA8D6"   # lighter steel blue — secondary accent / hover
ACCENT     = "#7FA8D6"
GOOD       = "#4FAE81"   # muted green — positive
BAD        = "#D9795B"   # muted terracotta — negative / risk
WARN       = "#D1A63F"   # muted gold — caution

CHART_COLORWAY = [PRIMARY_L, PRIMARY, "#8FA3BF", WARN, "#5E6C82"]

def plotly_theme(fig, height=440, legend_top=True):
    """Apply a consistent, clean dark theme to a plotly figure."""
    fig.update_layout(
        height=height,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORWAY,
        margin=dict(l=10, r=10, t=50 if legend_top else 30, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
        ) if legend_top else dict(),
        hoverlabel=dict(bgcolor=SURFACE_2, font_size=13, font_family="Inter, sans-serif",
                         font_color=INK, bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=BORDER, zeroline=False,
                      color=SUBTLE, tickfont=dict(color=SUBTLE))
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, showline=False,
                      color=SUBTLE, tickfont=dict(color=SUBTLE))
    return fig

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }}

    .stApp {{
        background: {CANVAS};
    }}

    /* ---- kill default streamlit chrome we don't want ---- */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    div.block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    /* ---- base text colors across the app ---- */
    p, span, label, li, div, .stMarkdown {{
        color: {INK};
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {SUBTLE} !important;
    }}
    code {{
        background: {SURFACE_2} !important;
        color: {PRIMARY_L} !important;
        border: 1px solid {BORDER};
    }}
    a, a:visited {{ color: {PRIMARY_L} !important; }}

    /* ---- hero header (slim band) ---- */
    .rf-hero {{
        background: linear-gradient(115deg, {PRIMARY_D} 0%, {PRIMARY} 100%);
        border-radius: 14px;
        padding: 0.9rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 22px -14px rgba(0,0,0,0.65);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.7rem 1.5rem;
    }}
    .rf-hero-left {{
        flex: 1 1 340px;
        min-width: 0;
    }}
    .rf-hero-kicker {{
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.62rem;
        font-weight: 700;
        opacity: 0.75;
        margin-bottom: 0.1rem;
    }}
    .rf-hero h1 {{
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
        color: white;
        display: inline;
    }}
    .rf-hero p {{
        font-size: 0.8rem;
        opacity: 0.85;
        max-width: 620px;
        margin: 0.15rem 0 0 0;
        line-height: 1.4;
    }}
    .rf-hero-tags {{
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        flex: 0 0 auto;
    }}
    .rf-tag {{
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.26);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        white-space: nowrap;
    }}

    /* ---- section headers ---- */
    .rf-section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {INK};
        margin: 0.2rem 0 0.2rem 0;
        letter-spacing: -0.01em;
    }}
    .rf-section-sub {{
        color: {SUBTLE};
        font-size: 0.9rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }}

    /* ---- metric cards ---- */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
        box-shadow: 0 2px 10px -6px rgba(0,0,0,0.5);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {SUBTLE} !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {SUBTLE} !important; }}
    div[data-testid="stMetricValue"] {{
        color: {INK} !important;
        font-weight: 800;
    }}
    div[data-testid="stMetricDelta"] svg {{ display: inline; }}

    /* ---- generic card ---- */
    .rf-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 2px 10px -6px rgba(0,0,0,0.5);
        height: 100%;
    }}
    .rf-card h4 {{
        margin: 0 0 0.7rem 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: {INK};
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .rf-stat-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.42rem 0;
        border-bottom: 1px dashed {BORDER};
        font-size: 0.88rem;
        color: {INK};
    }}
    .rf-stat-row:last-child {{ border-bottom: none; }}
    .rf-stat-row span:first-child {{ color: {SUBTLE}; }}
    .rf-stat-row b {{ font-weight: 700; }}

    .rf-pill {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
    }}
    .rf-pill-good {{ background: rgba(79,174,129,0.16); color: {GOOD}; }}
    .rf-pill-bad  {{ background: rgba(217,121,91,0.16); color: {BAD}; }}
    .rf-pill-warn {{ background: rgba(209,166,63,0.16); color: {WARN}; }}
    .rf-pill-info {{ background: rgba(126,168,214,0.16); color: {PRIMARY_L}; }}

    /* ---- recommendation / callout boxes ---- */
    .rf-callout {{
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        font-size: 0.88rem;
        line-height: 1.55;
        border: 1px solid;
        margin-top: 0.6rem;
    }}
    .rf-callout-reco {{
        background: rgba(126,168,214,0.08);
        border-color: rgba(126,168,214,0.28);
        color: {INK};
    }}
    .rf-callout-warn {{
        background: rgba(209,166,63,0.08);
        border-color: rgba(209,166,63,0.3);
        color: {INK};
    }}
    .rf-callout b {{ font-weight: 700; color: {PRIMARY_L}; }}
    .rf-callout-warn b {{ color: {WARN}; }}

    /* ---- tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 5px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 42px;
        border-radius: 8px;
        color: {SUBTLE} !important;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 16px;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: inherit !important; }}
    .stTabs [aria-selected="true"] {{
        background: {PRIMARY} !important;
        color: #FFFFFF !important;
    }}
    .stTabs [aria-selected="true"] p {{ color: #FFFFFF !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent !important; }}
    .stTabs [data-baseweb="tab-border"] {{ background-color: transparent !important; }}

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: #080B11;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: #C4CCDA !important;
    }}
    section[data-testid="stSidebar"] .rf-side-brand {{
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-bottom: 0.1rem;
    }}
    section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] a:visited {{
        color: {PRIMARY_L} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {BORDER};
    }}

    /* ---- dataframe ---- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ---- sliders ---- */
    div[data-testid="stSlider"] label p {{ color: {INK} !important; }}
    div[data-baseweb="slider"] [role="slider"] {{
        background-color: {PRIMARY_L} !important;
    }}

    /* ---- download button ---- */
    div.stDownloadButton button {{
        background: {PRIMARY};
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
    }}
    div.stDownloadButton button:hover {{
        background: {PRIMARY_D};
        color: white;
    }}
    div.stDownloadButton button p {{ color: white !important; }}

    /* ---- bordered containers (full report) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE};
        border-color: {BORDER} !important;
        border-radius: 14px;
    }}

    h1, h2, h3, h4, h5, h6 {{ color: {INK} !important; }}
    hr {{ border-color: {BORDER}; }}

    /* ---- equal-height metric boxes (native st.metric) ---- */
    div[data-testid="stMetric"] {{
        min-height: 126px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    /* keep rows of metrics stretched to their tallest sibling */
    div[data-testid="stHorizontalBlock"] {{
        align-items: stretch;
    }}
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] {{
        display: flex;
    }}
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] > div {{
        width: 100%;
    }}

    /* ---- KPI grid (overview top row) ---- */
    .rf-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
    }}
    .rf-kpi {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid var(--kpi-accent, {PRIMARY_L});
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 10px -6px rgba(0,0,0,0.5);
    }}
    .rf-kpi-label {{
        color: {SUBTLE};
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}
    .rf-kpi-value {{
        color: {INK};
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .rf-kpi-sub {{
        color: {SUBTLE};
        font-size: 0.76rem;
        margin-top: 0.3rem;
    }}
    @media (max-width: 900px) {{
        .rf-kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    /* ---- generic equal-height card grid (used for reco cards etc.) ---- */
    .rf-grid {{
        display: grid;
        gap: 0.85rem;
    }}
    .rf-grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .rf-grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
    @media (max-width: 900px) {{
        .rf-grid-3, .rf-grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .rf-grid .rf-card {{ height: 100%; }}

    /* ---- capability chips (overview "what this does") ---- */
    .rf-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.7rem;
    }}
    .rf-chip {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 9px;
        padding: 0.55rem 0.8rem;
        font-size: 0.82rem;
        color: {INK};
        flex: 1 1 190px;
    }}
    .rf-chip b {{ display: block; font-size: 0.72rem; color: {SUBTLE}; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.2rem; }}

    /* ---- sidebar tightened layout ---- */
    section[data-testid="stSidebar"] div.element-container {{
        margin-bottom: 0 !important;
    }}
    section[data-testid="stSidebar"] div.block-container {{
        padding-top: 1.2rem;
    }}
    .rf-side-block {{ margin-bottom: 1rem; }}
    .rf-side-label {{
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.68rem;
        font-weight: 700;
        color: #6B7688 !important;
        margin-bottom: 0.3rem;
    }}
    .rf-side-block p, .rf-side-block li {{
        font-size: 0.83rem;
        margin: 0.1rem 0;
        line-height: 1.4;
    }}
    .rf-side-block ul {{ margin: 0; padding-left: 1.1rem; }}
    .rf-side-links a {{
        display: block;
        font-size: 0.83rem;
        margin-bottom: 0.25rem;
    }}
    .rf-side-hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 0.8rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div class="rf-side-brand">📦 ReturnFlowAI</div>
    <div style="color:#8B96A8; font-size:0.82rem; margin-bottom:0.9rem;">Return forecasting &amp; business impact</div>
    <hr class="rf-side-hr">

    <div class="rf-side-block">
        <div class="rf-side-label">Dataset</div>
        <p>Olist Brazilian E-Commerce<br>99,441 orders · 85 weeks</p>
    </div>

    <div class="rf-side-block">
        <div class="rf-side-label">Key levers identified</div>
        <ul>
            <li>Delivery buffer</li>
            <li>Product category</li>
            <li>Seller quality</li>
        </ul>
    </div>

    <div class="rf-side-block rf-side-links">
        <div class="rf-side-label">Links</div>
        <a href="https://github.com/yashvardhanshah/ReturnFlow" target="_blank">🔗 GitHub Repository</a>
        <a href="https://returnflow.streamlit.app/" target="_blank">🌐 Live Dashboard</a>
    </div>

    <hr class="rf-side-hr">
    <div style="color:#5E6878; font-size:0.76rem;">Built by Yashvardhan Shah</div>
    """, unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================

st.markdown(f"""
<div class="rf-hero">
    <div class="rf-hero-left">
        <span class="rf-hero-kicker">E-COMMERCE ANALYTICS · OLIST DATASET</span><br>
        <h1>📦 ReturnFlowAI</h1>
        <p>Forecasting weekly returns, identifying return drivers, simulating delivery-speed impact, and flagging risky sellers.</p>
    </div>
    <div class="rf-hero-tags">
        <span class="rf-tag">XGBoost · 20.93% MAPE</span>
        <span class="rf-tag">SARIMA benchmark</span>
        <span class="rf-tag">Hypothesis testing</span>
        <span class="rf-tag">SQL seller analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_forecast, tab_causes, tab_ab, tab_sellers, tab_report = st.tabs(
    ["Overview", "Forecasting", "Root Cause", "A/B Simulator", "Seller Flags", "Full Report"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.markdown(f"""
    <div class="rf-kpi-grid">
        <div class="rf-kpi" style="--kpi-accent:{PRIMARY_L};">
            <div class="rf-kpi-label">Weeks Analyzed</div>
            <div class="rf-kpi-value">85</div>
        </div>
        <div class="rf-kpi" style="--kpi-accent:{PRIMARY_L};">
            <div class="rf-kpi-label">Orders Analyzed</div>
            <div class="rf-kpi-value">99,441</div>
        </div>
        <div class="rf-kpi" style="--kpi-accent:{GOOD};">
            <div class="rf-kpi-label">Best Forecast MAPE</div>
            <div class="rf-kpi-value">{XGB_MAPE}%</div>
            <div class="rf-kpi-sub">XGBoost</div>
        </div>
        <div class="rf-kpi" style="--kpi-accent:{WARN};">
            <div class="rf-kpi-label">Est. Weekly Returns Preventable</div>
            <div class="rf-kpi-value">~43</div>
            <div class="rf-kpi-sub">via delivery buffer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rf-card">
        <h4>🧭 What this project does</h4>
        <div class="rf-chip-row">
            <div class="rf-chip"><b>Forecasting</b>SARIMA vs. XGBoost, benchmarked head-to-head</div>
            <div class="rf-chip"><b>Root cause</b>Hypothesis testing on buffer &amp; category</div>
            <div class="rf-chip"><b>Simulation</b>Business impact of delivery reliability</div>
            <div class="rf-chip"><b>Seller analytics</b>SQL window functions flag risk accounts</div>
            <div class="rf-chip"><b>Output</b>Ranked, concrete recommendations</div>
        </div>
        <div class="rf-callout rf-callout-warn" style="margin-top:1rem;">
            <b>Scope note:</b> this dataset has no explicit "returned" flag. All return-related findings use a
            <b style="color:{WARN};">return-proxy</b> (1–2 star review or cancelled status) as a stand-in for
            genuine returns. Treat figures as directional, decision-grade estimates — not audited numbers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="rf-section-title">Ranked recommendations</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rf-grid rf-grid-3">
        <div class="rf-card">
            <span class="rf-pill rf-pill-good">FASTEST</span>
            <h4 style="margin-top:0.6rem;">1 · Review flagged sellers</h4>
            <p style="font-size:0.85rem; color:{SUBTLE}; margin:0;">Smallest effort, fastest payoff — a short list of named accounts, not a policy change.</p>
        </div>
        <div class="rf-card">
            <span class="rf-pill rf-pill-warn">HIGHEST IMPACT</span>
            <h4 style="margin-top:0.6rem;">2 · Pilot delivery-speed fix</h4>
            <p style="font-size:0.85rem; color:{SUBTLE}; margin:0;">~43 fewer returns/week estimated — needs a real pilot to confirm before scaling.</p>
        </div>
        <div class="rf-card">
            <span class="rf-pill rf-pill-info">OPERATIONAL</span>
            <h4 style="margin-top:0.6rem;">3 · Adopt XGBoost forecasting</h4>
            <p style="font-size:0.85rem; color:{SUBTLE}; margin:0;">Turns reverse-logistics staffing from reactive to planned.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FORECASTING ----------------
with tab_forecast:
    st.markdown('<div class="rf-section-title">SARIMA vs. XGBoost — Weekly Return Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-section-sub">Test period: June – August 2018 · both models trained only on the preceding 73 weeks</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=FORECAST_DF["week"], y=FORECAST_DF["Actual"],
                              mode="lines+markers", name="Actual",
                              line=dict(width=3, color=INK), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=FORECAST_DF["week"], y=FORECAST_DF["SARIMA"],
                              mode="lines+markers", name="SARIMA",
                              line=dict(width=2, dash="dot", color=WARN)))
    fig.add_trace(go.Scatter(x=FORECAST_DF["week"], y=FORECAST_DF["XGBoost"],
                              mode="lines+markers", name="XGBoost",
                              line=dict(width=2.5, color=ACCENT)))
    fig.update_layout(xaxis_title="Week", yaxis_title="Return-Proxy Count")
    plotly_theme(fig, height=440)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="rf-card"><h4>📉 SARIMA (walk-forward, spike flag, trend)</h4>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        sc1.metric("MAPE", f"{SARIMA_MAPE}%")
        sc2.metric("RMSE", f"{SARIMA_RMSE}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="rf-card"><h4>🏆 XGBoost (lag + calendar features)</h4>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        sc1.metric("MAPE", f"{XGB_MAPE}%", delta=f"-{SARIMA_MAPE - XGB_MAPE:.2f} pts vs SARIMA", delta_color="normal")
        sc2.metric("RMSE", f"{XGB_RMSE}", delta=f"-{SARIMA_RMSE - XGB_RMSE:.2f} vs SARIMA", delta_color="normal")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="rf-callout rf-callout-reco">
        <b>Recommendation:</b> deploy XGBoost as the primary weekly forecast for reverse-logistics
        staffing and budget planning, with SARIMA retained as a secondary sanity-check model.
    </div>
    """, unsafe_allow_html=True)

# ---------------- ROOT CAUSE ----------------
with tab_causes:
    st.markdown('<div class="rf-section-title">Return Rate by Product Category</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-section-sub">Highest vs. lowest five categories, relative to the platform-wide average</div>', unsafe_allow_html=True)

    fig2 = px.bar(CATEGORY_DF.sort_values("return_rate"), x="return_rate", y="category",
                   orientation="h", color="group",
                   color_discrete_map={"Highest 5": BAD, "Lowest 5": GOOD})
    fig2.add_vline(x=OVERALL_AVG_RETURN_RATE, line_dash="dash", line_color=SUBTLE,
                    annotation_text="Overall avg (~14.1%)", annotation_font_color=SUBTLE)
    fig2.update_layout(xaxis_title="Return Rate", yaxis_title="", xaxis_tickformat=".0%", legend_title="")
    plotly_theme(fig2, height=440)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="rf-section-title" style="font-size:1.05rem; margin-top:1rem;">Statistical findings</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="rf-card">
            <h4>📦 Delivery buffer <span class="rf-pill rf-pill-info">Welch's t-test</span></h4>
            <div class="rf-stat-row"><span>Non-returned orders</span><b>-12.86 days avg buffer</b></div>
            <div class="rf-stat-row"><span>Returned orders</span><b>-5.13 days avg buffer</b></div>
            <div class="rf-stat-row"><span>Test statistic</span><b>t = 54.17, p &lt; 0.000001</b></div>
            <div class="rf-stat-row"><span>Consistency</span><b>Held across all 26 categories</b></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="rf-card">
            <h4>🏷️ Product category <span class="rf-pill rf-pill-info">Chi-square test</span></h4>
            <div class="rf-stat-row"><span>Test statistic</span><b>χ² = 299.41, p &lt; 0.000001</b></div>
            <div class="rf-stat-row"><span>Highest</span><b>moveis_escritorio — 22.5%</b></div>
            <div class="rf-stat-row"><span>Lowest</span><b>livros_interesse_geral — 8.3%</b></div>
            <div class="rf-stat-row"><span>Degrees of freedom</span><b>25</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rf-callout rf-callout-reco">
        <b>Recommendation:</b> prioritize office furniture and home-decor categories for quality
        review — they sit well above the ~14% platform average return rate.
    </div>
    """, unsafe_allow_html=True)

# ---------------- A/B SIMULATOR ----------------
with tab_ab:
    st.markdown('<div class="rf-section-title">Delivery-Speed A/B Test &amp; Impact Simulator</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="rf-section-sub">
    Baseline: <b style="color:{INK};">{LOW_BUFFER_ORDERS:,}</b> low-buffer orders return at
    <b style="color:{BAD};">{LOW_BUFFER_RATE:.1%}</b>, vs.
    <b style="color:{INK};">{HIGH_BUFFER_ORDERS:,}</b> high-buffer orders at
    <b style="color:{GOOD};">{HIGH_BUFFER_RATE:.1%}</b>
    (χ² = 1515.04, p &lt; 0.000001).
    </div>
    """, unsafe_allow_html=True)

    # ---- Actual notebook results (Phase 5) ----
    st.markdown('<div class="rf-section-title" style="font-size:1.05rem; margin-top:0.4rem;">Actual A/B Test Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rf-section-sub">Direct output from the analysis notebook — no slider inputs, these are the measured results.</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("Actual Returns (Low-Buffer Group)", "7,589")
    a2.metric("Simulated Returns (if Matched High-Buffer Rate)", "3,908")
    a3.metric("Returns Prevented", "3,681", delta="-48.5% relative reduction", delta_color="inverse")

    b1, b2 = st.columns(2)
    b1.metric("Relative Reduction", "48.5%")
    b2.metric("Returns Prevented / Week (Platform-Wide)", "43.3")

    st.markdown(f"""
    <div class="rf-card" style="margin-top:0.8rem;">
        <h4>📋 Summary</h4>
        <div class="rf-stat-row"><span>Grouping</span><b>High-buffer (52,589 orders) vs. low-buffer (43,887 orders), split on the median delivery delay as a natural cutoff</b></div>
        <div class="rf-stat-row"><span>Gap found</span><b>8.90% vs. 17.29% return rate — statistically significant (χ², p &lt; 0.001)</b></div>
        <div class="rf-stat-row"><span>Simulated impact</span><b>~48.5% relative reduction, ~43 fewer returns/week if low-buffer orders matched high-buffer performance</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rf-callout rf-callout-reco" style="margin-top:0.9rem;">
        <b>This fulfills:</b> A/B simulation + business impact translation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.8rem 0 1.4rem 0;'>", unsafe_allow_html=True)

    # ---- Interactive simulator ----
    st.markdown('<div class="rf-section-title" style="font-size:1.05rem;">Interactive Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-section-sub">Adjust the sliders below to model a real intervention at a different rollout percentage or cost assumption.</div>', unsafe_allow_html=True)

    st.markdown('<div class="rf-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        pct_improved = st.slider(
            "% of low-buffer orders improved to high-buffer performance",
            min_value=0, max_value=100, value=100, step=5
        )
    with col_b:
        cost_per_return = st.slider(
            "Estimated cost per prevented return ($)",
            min_value=1, max_value=50, value=10, step=1
        )
    st.markdown('</div>', unsafe_allow_html=True)

    pct = pct_improved / 100
    new_rate = pct * HIGH_BUFFER_RATE + (1 - pct) * LOW_BUFFER_RATE
    current_returns = LOW_BUFFER_ORDERS * LOW_BUFFER_RATE
    new_returns = LOW_BUFFER_ORDERS * new_rate
    returns_prevented = current_returns - new_returns
    weekly_prevented = returns_prevented / TOTAL_WEEKS
    annual_savings = weekly_prevented * 52 * cost_per_return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="rf-section-title" style="font-size:1.05rem;">Projected new impact (at selected inputs)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("New Low-Buffer Return Rate", f"{new_rate:.1%}",
              delta=f"-{(LOW_BUFFER_RATE - new_rate):.1%}", delta_color="inverse")
    m2.metric("Returns Prevented (total)", f"{returns_prevented:,.0f}")
    m3.metric("Returns Prevented / Week", f"{weekly_prevented:.1f}")
    m4.metric("Est. Annual Savings", f"${annual_savings:,.0f}")

    fig3 = go.Figure(data=[
        go.Bar(name="Before", x=["Low-Buffer Group — Avg Returns / Week"],
               y=[current_returns / TOTAL_WEEKS], marker_color=BAD, width=0.35),
        go.Bar(name="After (projected)", x=["Low-Buffer Group — Avg Returns / Week"],
               y=[new_returns / TOTAL_WEEKS], marker_color=GOOD, width=0.35),
    ])
    fig3.update_layout(yaxis_title="Avg Returns / Week", barmode="group")
    plotly_theme(fig3, height=380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="rf-callout rf-callout-warn">
        <b>Caveat:</b> this is a simulated, correlational estimate based on historical order groups,
        not a randomized controlled experiment. Validate with a real pilot before using in a formal budget.
    </div>
    """, unsafe_allow_html=True)

# ---------------- SELLER FLAGS ----------------
with tab_sellers:
    st.markdown('<div class="rf-section-title">Sellers Flagged for Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-section-sub">Sellers with 15+ items sold and a return rate significantly above their category average — identified via SQL window functions (<code>AVG() OVER</code>, <code>RANK() OVER</code>).</div>', unsafe_allow_html=True)

    top1, top2, top3 = st.columns(3)
    worst = SELLER_DF.loc[SELLER_DF["seller_return_rate"].idxmax()]
    biggest_gap = SELLER_DF.loc[SELLER_DF["gap_vs_category"].idxmax()]
    top1.metric("Sellers Flagged", f"{len(SELLER_DF)}")
    top2.metric("Highest Seller Return Rate", f"{worst['seller_return_rate']:.1%}", worst["category"])
    top3.metric("Largest Gap vs. Category Avg", f"+{biggest_gap['gap_vs_category']:.1%}", biggest_gap["category"])

    st.markdown("<br>", unsafe_allow_html=True)

    display_df = SELLER_DF.copy().sort_values("gap_vs_category", ascending=False)
    display_df["seller_return_rate"] = display_df["seller_return_rate"].map("{:.1%}".format)
    display_df["category_avg_return_rate"] = display_df["category_avg_return_rate"].map("{:.1%}".format)
    display_df["gap_vs_category"] = SELLER_DF.sort_values("gap_vs_category", ascending=False)["gap_vs_category"].map("+{:.1%}".format)
    display_df.columns = ["Seller ID", "Category", "Items Sold", "Seller Return Rate",
                           "Category Avg", "Gap vs. Category"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Seller ID": st.column_config.TextColumn(width="medium"),
        },
    )

    st.markdown("""
    <div class="rf-callout rf-callout-reco">
        <b>Recommendation:</b> route these accounts into an immediate quality review process —
        the fastest, cheapest fix available from this analysis.
    </div>
    """, unsafe_allow_html=True)

# ---------------- FULL REPORT ----------------
with tab_report:
    st.markdown('<div class="rf-section-title">Full Business Impact Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="rf-section-sub">The complete write-up behind this dashboard, including methodology, caveats, and ranked recommendations.</div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇ Download Full Report (Markdown)",
        data=REPORT_TEXT,
        file_name="ReturnFlowAI_Business_Report.md",
        mime="text/markdown",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(REPORT_TEXT)