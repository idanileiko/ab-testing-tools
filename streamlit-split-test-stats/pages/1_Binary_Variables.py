import streamlit as st

st.set_page_config(
    page_title="A/B Testing Analysis for Binary Variables",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 A/B Testing Statistical Analysis")
st.write("Upload your experiment data and run statistical tests between groups.")
st.write("Note: currently only analyzes binary metrics (e.g. conversion rates, click rates, etc). A future version will also run stats analyses on continuous variables like amount spent.")
