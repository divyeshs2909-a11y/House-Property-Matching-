import streamlit as st
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()
st.set_page_config(page_title="Property Match Score (Embeddings)", layout="wide")
st.title("🏠 Property Match Score (Semantic Embeddings)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("User Preferences")
    user_id = st.text_input("User ID", "U1")
    budget = st.number_input("Budget", min_value=0.0, value=8000000.0, step=50000.0)
    u_bed = st.number_input("Bedrooms (needed)", min_value=0.0, value=2.0, step=1.0)
    u_bath = st.number_input("Bathrooms (needed)", min_value=0.0, value=2.0, step=1.0)
    u_desc = st.text_area("Qualitative Description (user)", "Cozy, warm home near parks, great natural light")

with col2:
    st.subheader("Property Characteristics")
    prop_id = st.text_input("Property ID", "P101")
    price = st.number_input("Price", min_value=0.0, value=7900000.0, step=50000.0)
    p_bed = st.number_input("Bedrooms (property)", min_value=0.0, value=2.0, step=1.0)
    p_bath = st.number_input("Bathrooms (property)", min_value=0.0, value=2.0, step=1.0)
    p_desc = st.text_area("Qualitative Description (property)", "Warm and inviting retreat with abundant sunlight and nearby green spaces")

st.divider()
api_url = st.text_input("API URL", "http://127.0.0.1:8001/match")
backend = st.selectbox("Text backend", ["sbert", "gemini"], index=0)

if st.button("Calculate Match Score"):
    payload = {
        "user": {"User_ID": user_id, "Budget": budget, "Bedrooms": u_bed, "Bathrooms": u_bath, "Qualitative_Description": u_desc},
        "property": {"Property_ID": prop_id, "Price": price, "Bedrooms": p_bed, "Bathrooms": p_bath, "Qualitative_Description": p_desc},
        "text_backend": backend,
    }
    try:
        r = requests.post(api_url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.success(f"Match Score: **{data['match_score']} / 100**")
        st.json(data)
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.info("Start API: uvicorn src.api:app --port 8001 --workers 4")
