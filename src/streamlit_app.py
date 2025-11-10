# src/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/explain"

st.set_page_config(page_title="Telugu Code Explainer", layout="wide")
st.title("💡 Telugu Code Explainer")

# --- Input Section ---
code_input = st.text_area(
    "Paste your code here:",
    height=200,
    placeholder="def factorial(n): return 1 if n==0 else n*factorial(n-1)"
)

mode = st.selectbox("Explanation Mode", ["simple"], index=0)

# --- Action ---
if st.button("🔍 Explain in Telugu"):
    if not code_input.strip():
        st.warning("Please enter some code first.")
    else:
        with st.spinner("Generating explanation..."):
            response = requests.post(API_URL, json={"code": code_input, "mode": mode})

            if response.status_code == 200:
                data = response.json()

                # Create two equal columns for bilingual view
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📝 English Explanation")
                    st.markdown(
                        f"""
                        <div style='
                            background-color:#1e1e1e;
                            color:#f5f5f5;
                            padding:15px;
                            border-radius:10px;
                            font-size:16px;
                            line-height:1.7;
                            white-space:pre-wrap;
                            font-family: "Fira Code", monospace;
                        '>{data.get("english", "N/A")}</div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown("### 🌐 Telugu Explanation")
                    st.markdown(
                        f"""
                        <div style='
                            background-color:#1e1e1e;
                            color:#f5f5f5;
                            padding:15px;
                            border-radius:10px;
                            font-size:16px;
                            line-height:1.8;
                            white-space:pre-wrap;
                            font-family: "Noto Sans Telugu", sans-serif;
                        '>{data.get("telugu", "N/A")}</div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.error(f"Error {response.status_code}: {response.text}")
