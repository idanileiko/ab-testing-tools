import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Welcome!
    **Select a page from the sidebar** that fits your stats use-case!
"""
)
