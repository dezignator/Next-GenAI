"""
scripts/explain_dashboard.py
============================
Run with: streamlit run scripts/explain_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Next-GenAI Threat Intel", layout="wide")
st.title("🛡️ Next-GenAI: DoH Threat Intelligence & Explainability")

# Mock data loading (in production, load results/evaluation_report.json)
@st.cache_data
def load_data():
    # Simulated SHAP outputs
    return pd.DataFrame({
        "Feature": ["query_entropy", "sni_mismatch", "response_latency_ms", "cert_validity_days", "burstiness"],
        "SHAP Value": [0.45, 0.30, 0.15, 0.08, -0.02]
    })

df = load_data()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Alert Details")
    st.metric(label="Verdict", value="MALICIOUS", delta="High Confidence")
    st.metric(label="Confidence Score", value="98.4%")
    st.info("Attack Type Detected: **DNS Tunneling** (Eq. 3 Evasion Passed)")

with col2:
    st.subheader("SHAP Feature Attribution (Eq. 16)")
    fig = px.bar(df, x="SHAP Value", y="Feature", orientation='h',
                 color="SHAP Value", color_continuous_scale="RdBu",
                 title="Why did the model flag this?")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("System Health (Streaming)")
st.success("Kafka Pipeline: ✅ Active | Flink Workers: 4 | Latency: 42ms")
