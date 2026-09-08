import streamlit as st

st.set_page_config(page_title="ReturnFlowAI", layout="wide")

st.title("ReturnFlowAI")
st.subheader("E-Commerce Return Forecasting & Business Impact System")

st.markdown("""
This dashboard is under active development.

**Planned features:**
- Weekly return volume forecasting (SARIMA vs XGBoost)
- Root-cause analysis of return drivers (delivery delay, product category)
- Simulated A/B evaluation of delivery-speed impact on returns
- SQL-based seller and category return-risk analytics
""")

st.info("Forecast dashboard coming soon — check back for updates.")