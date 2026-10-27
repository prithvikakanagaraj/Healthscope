"""HealthScope entry point: page config, styling and navigation."""
import streamlit as st

from src.config import APP_NAME, TAGLINE
from src.ui import inject_css

st.set_page_config(page_title=f"{APP_NAME} - {TAGLINE}", page_icon="🩺", layout="wide")
inject_css()

pages = [
    st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Analytics_Dashboard.py", title="Analytics Dashboard", icon="📊"),
    st.Page("pages/3_Product_Explorer.py", title="Product Explorer", icon="🔎"),
    st.Page("pages/4_Review_Sentiment.py", title="Review Sentiment", icon="💬"),
    st.Page("pages/5_Big_Data_Insights.py", title="Big Data Insights", icon="💡"),
    st.Page("pages/6_Dataset_Manager.py", title="Dataset Manager", icon="🗂️"),
    st.Page("pages/7_About_Project.py", title="About the Project", icon="ℹ️"),
]
st.navigation(pages).run()
