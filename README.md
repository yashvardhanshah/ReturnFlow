<div align="center">

# 📦 ReturnFlowAI

### Forecast returns before they happen. Find out why they happen. Fix it fast.

**An end-to-end return-intelligence platform built on 99,441 real e-commerce orders —**
**forecasting, root-cause analysis, business-impact simulation, and seller risk flagging, in one dashboard.**

[![Live Dashboard](https://img.shields.io/badge/🌐_Live_Demo-returnflow.streamlit.app-3E6FA5?style=for-the-badge)](https://returnflow.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](#license)

**[🚀 Try the Live Dashboard](https://returnflow.streamlit.app/) &nbsp;•&nbsp; [📊 Explore the Code](https://github.com/yashvardhanshah/ReturnFlow)**

</div>

---

## ⚡ Why this project exists

Every return costs money twice — once shipping it out, again reversing it. Most teams find out returns spiked *after* the invoice lands. **ReturnFlowAI turns that around**: it forecasts weekly return volume, pinpoints the exact operational levers driving returns, and estimates the dollar impact of fixing them — all from a single, real 85-week e-commerce dataset.

> 💡 **TL;DR:** XGBoost forecasts weekly returns at **20.93% MAPE**. Tighter delivery buffers **nearly double** the return rate (17.3% vs 8.9%). A handful of sellers return **2–5× their category average**. This dashboard shows you exactly where, and estimates exactly how much closing the gap is worth.

---

## 🎯 Key Results at a Glance

| 🔍 Finding | 📈 Result |
|---|---|
| **Forecasting accuracy** | XGBoost: **20.93% MAPE**, 49.57 RMSE — beats SARIMA's 29.11% MAPE |
| **Delivery buffer effect** | Low-buffer orders return at **17.29%** vs **8.90%** for high-buffer orders (χ² = 1515.04, p < 0.000001) |
| **Estimated weekly upside** | **~43 fewer returns/week** if the delivery-buffer gap is closed |
| **Category risk** | Office furniture returns at **22.5%** — 2.7× the rate of general books (8.3%) |
| **Seller risk** | Flagged sellers return at **up to 90.0%**, vs **~18%** category average |

---

## 🧭 What's Inside

The dashboard is organized into six focused tabs:

| Tab | What it shows |
|---|---|
| 🏠 **Overview** | KPI snapshot + ranked, prioritized recommendations |
| 📉 **Forecasting** | SARIMA vs. XGBoost, head-to-head, on 12 held-out weeks |
| 🔬 **Root Cause** | Hypothesis testing — delivery buffer (t-test) & category (χ²) |
| 🧪 **A/B Simulator** | Real notebook results **+** an interactive slider to model your own rollout & cost assumptions |
| 🚩 **Seller Flags** | SQL window-function output naming specific high-risk sellers |
| 📄 **Full Report** | Complete write-up — methodology, caveats, and recommendations — downloadable as **Markdown or PDF** |

---

## 🛠️ Under the Hood

**Forecasting**
- `SARIMA` with walk-forward validation, spike-week flag, and trend component
- `XGBoost` with lag features, rolling means, and calendar seasonality
- Evaluated on identical, unseen 12-week test windows

**Statistical Testing**
- Welch's *t*-test on delivery buffer (returned vs. non-returned orders)
- Chi-square test of independence across 26 product categories
- Retrospective A/B comparison with a simulated impact projection

**Data Engineering**
- SQL window functions (`AVG() OVER`, `RANK() OVER`) against a local SQLite store of 114,092 order-items
- Return-proxy methodology (1–2★ review or cancellation) clearly flagged throughout as an approximation, not ground truth

**Stack:** `Python` · `pandas` · `XGBoost` · `statsmodels` (SARIMA) · `SQLite` · `Plotly` · `Streamlit` · `fpdf2`

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/yashvardhanshah/ReturnFlow.git
cd ReturnFlow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501` — no API keys, no external services, no setup beyond `pip install`.

---

## 📁 Repo Structure

```
ReturnFlow/
├── app.py                          # Streamlit dashboard (this project)
├── notebooks/                      # 01–06: forecasting, hypothesis testing, A/B sim, SQL
├── sql/
│   └── seller_category_analysis.sql
├── requirements.txt
└── README.md
```

---

## ⚠️ Honest Limitations

This project is built to be **decision-grade, not audited**:
- Returns are approximated via a review/cancellation **proxy** — the dataset has no explicit "returned" flag.
- Delivery-buffer and A/B findings are **observational associations**, not causal proof — a real pilot is the recommended next step before committing budget.
- Data spans Oct 2016–Aug 2018; models would need retraining on current data for production use.

Full caveats are laid out in-app under **Full Report**.

---

## 🗺️ Roadmap Ideas

- [ ] Swap review/cancellation proxy for an authoritative return flag, if/when available
- [ ] Live A/B pilot to convert the ~48.5% impact estimate into a confirmed figure
- [ ] Automate the seller-risk SQL query as a scheduled monthly job
- [ ] Add confidence intervals to the XGBoost forecast

---

## 🤝 Contributing

Issues and PRs are welcome — especially around forecasting improvements, additional causal-inference checks, or extending the seller-risk methodology.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

**Built by [Yashvardhan Shah](https://github.com/yashvardhanshah)**

⭐ If this project helped you think about returns differently, consider starring the repo!

</div>