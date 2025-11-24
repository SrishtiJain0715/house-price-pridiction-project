# app.py
# House Price App — Dark Mode + SHAP + EMI + Chatbot + Document Summary + Price Prediction + Compare + Insights
# Save this file next to your model.pkl

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import re
import io

# Optional SHAP import (only used if available)
try:
    import shap
except Exception:
    shap = None

# ---------------------------
# Page / UI theme (Dark)
# ---------------------------
st.set_page_config(page_title="House Price App", layout="wide", page_icon="🏡")
st.markdown(
    """
    <style>
    body, .stApp { background-color: #0e1117; color: #e6eef8; }
    .stSidebar { background-color: #161a23 !important; color: #e6eef8; }
    h1, h2, h3, h4, h5, h6 { color: #4DA8DA !important; }
    .stButton>button { background-color: #4DA8DA; color: black; border-radius: 6px; padding: 8px 14px; }
    .chat-bubble-user { background:#4DA8DA; padding:10px; border-radius:10px; margin:6px; width:fit-content; color:black; }
    .chat-bubble-bot { background:#22262d; padding:10px; border-radius:10px; margin:6px; width:fit-content; color:#e6eef8; }
    .stDownloadButton>button { background-color: #4DA8DA; color: black; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load model (safely)
# ---------------------------
model = None
model_loaded = False
if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"⚠ Error loading model.pkl: {e}")
else:
    st.sidebar.error("⚠ model.pkl missing! Place it in the same folder as app.py")

# ---------------------------
# Sidebar navigation
# ---------------------------
page = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "Price Prediction",
        "Compare Houses",
        "Document Summary + Prediction",
        "Insights",
        "Explainability (SHAP)",
        "EMI Calculator",
        "Chatbot Assistant",
    ],
)

# ---------------------------
# Utility: small extractive summarizer
# ---------------------------
def summarize_text(text, max_sentences=3):
    """
    Simple extractive summarizer: score sentences by word frequency.
    Returns top max_sentences sentences in original order.
    """
    # Split into sentences (naive)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    # Tokenize and compute word frequencies
    words = re.findall(r'\w+', text.lower())
    stopwords = set([
        # small stopword list
        'the','and','is','in','to','of','a','for','with','on','that','this','it','as','are','was','by','an','be','or'
    ])
    freqs = {}
    for w in words:
        if w in stopwords or w.isdigit():
            continue
        freqs[w] = freqs.get(w, 0) + 1
    # Score sentences
    sent_scores = []
    for s in sentences:
        s_words = re.findall(r'\w+', s.lower())
        score = sum(freqs.get(w,0) for w in s_words)
        sent_scores.append((s, score))
    # pick top sentences
    top = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    # keep original order
    top_set = set([t[0] for t in top])
    summary_sentences = [s for s in sentences if s in top_set]
    return " ".join(summary_sentences)

# ---------------------------
# Utility: extract numeric features from text
# ---------------------------
def extract_features_from_text(text):
    """
    Tries to extract: bedrooms, bathrooms, livingarea, condition, schools
    Returns dict with possible None values.
    """
    res = {"bedrooms": None, "bathrooms": None, "livingarea": None, "condition": None, "schools": None}
    t = text.lower()

    # bedrooms: look for patterns like '3 bedroom', '3 bedrooms' or 'bedrooms: 3'
    m = re.search(r'(\d+)\s*(?:bedroom|bedrooms|br\b)', t)
    if m:
        res["bedrooms"] = int(m.group(1))

    # bathrooms
    m = re.search(r'(\d+)\s*(?:bathroom|bathrooms|ba\b)', t)
    if m:
        res["bathrooms"] = int(m.group(1))

    # living area: patterns like '2000 sq ft', '2000 sqft', 'living area 2000'
    m = re.search(r'(\d{3,6})\s*(?:sq\s*ft|sqft|sq\.ft|sqm|m2|sq m|square foot|sq ft)', t)
    if not m:
        # sometimes 'living area 2000'
        m = re.search(r'(?:living area|area)\s*[:\-]?\s*(\d{3,6})', t)
    if m:
        res["livingarea"] = int(m.group(1))

    # condition: look for words like excellent/good/fair/1-5
    # If numeric rating present:
    m = re.search(r'condition\s*[:\-]?\s*(\d)', t)
    if m:
        res["condition"] = int(m.group(1))
    else:
        if "excellent" in t:
            res["condition"] = 5
        elif "very good" in t or "very good condition" in t:
            res["condition"] = 4
        elif "good condition" in t or "good condition" in t:
            res["condition"] = 4
        elif "fair" in t:
            res["condition"] = 3
        elif "poor" in t:
            res["condition"] = 1

    # schools nearby:
    m = re.search(r'(\d+)\s*(?:school|schools|colleges|institutes)\b', t)
    if m:
        res["schools"] = int(m.group(1))
    else:
        # presence words may indicate nearby but no count; default to 1 if 'near school' found
        if "near school" in t or "nearby school" in t or "close to school" in t:
            res["schools"] = 1

    return res

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.title("🏡 Advanced House Price Prediction System")
    st.subheader("AI-powered • Dark Theme • Document Summary + Price Prediction")

    st.write("---")
    st.markdown(
        """
        *What you can do in this project*
        - Predict price from user inputs.
        - Upload or paste a property description (pdf/txt) → get a summary and automatic feature extraction.
        - Edit extracted features (if any extraction is wrong) and predict price.
        - Compare two houses.
        - SHAP explainability (if shap installed & model compatible).
        - EMI calculator for loan planning.
        - Interactive chatbot for quick help.
        """
    )
    st.write("---")
    st.info("Tip: Use *Document Summary + Prediction* from sidebar to upload a listing or description and get a summary + price.")

# ---------------------------
# PRICE PREDICTION (manual inputs)
# ---------------------------
elif page == "Price Prediction":
    st.title("📈 Predict House Price (Manual Inputs)")

    if not model_loaded:
        st.error("Model not loaded, prediction disabled.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            bedrooms = st.number_input("🛏 Bedrooms", min_value=0, value=3)
            bathrooms = st.number_input("🛁 Bathrooms", min_value=0, value=2)
            condition = st.number_input("📊 Condition (1-5)", min_value=1, max_value=5, value=3)
        with col2:
            livingarea = st.number_input("📐 Living Area (sq ft)", min_value=50, value=2000)
            schools = st.number_input("🏫 Schools Nearby", min_value=0, value=2)

        if st.button("Predict Price"):
            X = np.array([[bedrooms, bathrooms, livingarea, condition, schools]])
            try:
                price = model.predict(X)[0]
                st.success(f"💰 Estimated Price: *${price:,.2f}*")
                fig, ax = plt.subplots()
                ax.bar(["Predicted Price"], [price])
                ax.set_ylabel("Price ($)")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------
# COMPARE HOUSES
# ---------------------------
elif page == "Compare Houses":
    st.title("🏘 Compare Two Houses")

    if not model_loaded:
        st.error("Model not loaded, comparison disabled.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("House A")
            A_bed = st.number_input("Bedrooms (A)", 0, value=3, key="A_bed")
            A_bath = st.number_input("Bathrooms (A)", 0, value=2, key="A_bath")
            A_area = st.number_input("Living Area (A)", 200, value=1800, key="A_area")
            A_cond = st.number_input("Condition (A)", 1, 5, value=3, key="A_cond")
            A_school = st.number_input("Schools Nearby (A)", 0, value=2, key="A_school")
        with colB:
            st.subheader("House B")
            B_bed = st.number_input("Bedrooms (B)", 0, value=4, key="B_bed")
            B_bath = st.number_input("Bathrooms (B)", 0, value=3, key="B_bath")
            B_area = st.number_input("Living Area (B)", 200, value=2400, key="B_area")
            B_cond = st.number_input("Condition (B)", 1, 5, value=4, key="B_cond")
            B_school = st.number_input("Schools Nearby (B)", 0, value=3, key="B_school")

        if st.button("Compare Prices"):
            try:
                priceA = model.predict([[A_bed, A_bath, A_area, A_cond, A_school]])[0]
                priceB = model.predict([[B_bed, B_bath, B_area, B_cond, B_school]])[0]
                st.success(f"House A: *${priceA:,.2f}*")
                st.success(f"House B: *${priceB:,.2f}*")
                diff = priceB - priceA
                pct = (diff / priceA * 100) if priceA else 0
                st.info(f"Difference: ${diff:,.2f} ({pct:,.1f}%)")
                fig, ax = plt.subplots()
                ax.bar(["House A", "House B"], [priceA, priceB])
                ax.set_ylabel("Price ($)")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------
# DOCUMENT SUMMARY + PREDICTION
# ---------------------------
elif page == "Document Summary + Prediction":
    st.title("📄 Document Summary + Price Prediction")
    st.write("Upload a property description (TXT or PDF) or paste text. App will summarize, try to extract features, and allow prediction.")

    uploaded_file = st.file_uploader("Upload .txt or .pdf (optional)", type=["txt", "pdf"])
    pasted = st.text_area("Or paste property description here", height=200)

    raw_text = ""
    if uploaded_file:
        fname = uploaded_file.name.lower()
        if fname.endswith(".txt"):
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            # use PyPDF2 if installed; otherwise ask user to paste text
            try:
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(uploaded_file)
                pages = []
                for p in pdf_reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                raw_text = "\n".join(pages)
            except Exception as e:
                st.warning("PDF reading requires PyPDF2. Please paste text or install PyPDF2.")
                raw_text = ""
    if pasted and not raw_text:
        raw_text = pasted

    if raw_text.strip() == "":
        st.info("Upload a document or paste text to get a summary and automatic feature extraction.")
    else:
        # 1) Summary
        st.subheader("🔍 Summary")
        summary = summarize_text(raw_text, max_sentences=4)
        st.write(summary)

        # 2) Try to extract features
        st.subheader("🧾 Extracted Features (auto)")
        extracted = extract_features_from_text(raw_text)
        st.write("Extraction may miss some values — edit below if needed.")

        # Provide editable inputs (pre-filled with extracted or default)
        col1, col2 = st.columns(2)
        with col1:
            bedrooms = st.number_input("🛏 Bedrooms", min_value=0, value=extracted["bedrooms"] if extracted["bedrooms"] is not None else 3, key="doc_bed")
            bathrooms = st.number_input("🛁 Bathrooms", min_value=0, value=extracted["bathrooms"] if extracted["bathrooms"] is not None else 2, key="doc_bath")
            condition = st.number_input("📊 Condition (1-5)", min_value=1, max_value=5, value=extracted["condition"] if extracted["condition"] is not None else 3, key="doc_cond")
        with col2:
            livingarea = st.number_input("📐 Living Area (sq ft)", min_value=50, value=extracted["livingarea"] if extracted["livingarea"] is not None else 2000, key="doc_area")
            schools = st.number_input("🏫 Schools Nearby", min_value=0, value=extracted["schools"] if extracted["schools"] is not None else 1, key="doc_sch")

        # Predict button
        if st.button("Predict from Document"):
            if not model_loaded:
                st.error("Model not loaded — cannot predict.")
            else:
                X = np.array([[bedrooms, bathrooms, livingarea, condition, schools]])
                try:
                    price = model.predict(X)[0]
                    st.success(f"💰 Predicted Price from Document: *${price:,.2f}*")
                    fig, ax = plt.subplots()
                    ax.bar(["Predicted Price"], [price])
                    ax.set_ylabel("Price ($)")
                    st.pyplot(fig)
                    # also show how features influenced (if shap available and model is tree-based)
                    if shap is not None:
                        try:
                            # use TreeExplainer if possible (works for RandomForest, etc.)
                            if hasattr(model, "best_estimator_"):
                                expl = shap.TreeExplainer(model.best_estimator_)
                            else:
                                expl = shap.TreeExplainer(model)
                            shap_vals = expl.shap_values(X)
                            st.subheader("SHAP (local) — feature contributions")
                            fig2 = plt.figure()
                            shap.force_plot(expl.expected_value, shap_vals, X, feature_names=["bedrooms","bathrooms","livingarea","condition","schools"], matplotlib=True, show=False)
                            st.pyplot(fig2)
                        except Exception:
                            # fallback: skip shap if any error
                            pass
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ---------------------------
# INSIGHTS
# ---------------------------
elif page == "Insights":
    st.title("📊 Market Insights")
    st.write("Some demo visualizations to show trends.")
    data = pd.DataFrame(
        {"Living Area": [800, 1200, 1500, 2000, 2500, 3000], "Price": [100000, 150000, 180000, 250000, 320000, 400000]}
    )
    fig, ax = plt.subplots()
    ax.plot(data["Living Area"], data["Price"], marker="o")
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Price ($)")
    ax.set_title("Living Area vs Price (Demo)")
    st.pyplot(fig)
    st.write("---")
    st.subheader("Key Insights")
    st.write(
        "- Larger living area tends to raise price.\n"
        "- Better condition adds premium value.\n"
        "- Proximity to schools increases demand."
    )

# ---------------------------
# SHAP Explainability page
# ---------------------------
elif page == "Explainability (SHAP)":
    st.title("🧠 SHAP Explainability")
    if shap is None:
        st.error("SHAP is not installed in this environment. To enable, install shap package.")
    elif not model_loaded:
        st.error("Model not loaded — cannot run SHAP.")
    else:
        st.write("Local SHAP examples (if model is tree-based).")
        try:
            # Choose a sample input to explain
            sample = np.array([[3, 2, 2000, 3, 2]])
            if hasattr(model, "best_estimator_"):
                explainer = shap.TreeExplainer(model.best_estimator_)
            else:
                explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample)
            st.subheader("SHAP bar plot")
            fig = plt.figure()
            shap.summary_plot(shap_vals, sample, feature_names=["bedrooms","bathrooms","livingarea","condition","schools"], plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP error: {e}")

# ---------------------------
# EMI Calculator
# ---------------------------
elif page == "EMI Calculator":
    st.title("💳 EMI Calculator")
    amount = st.number_input("Loan Amount", min_value=1000, value=500000)
    rate = st.number_input("Interest Rate % (Annual)", min_value=0.1, value=7.5)
    years = st.number_input("Loan Tenure (Years)", min_value=1, value=20)
    if st.button("Calculate EMI"):
        r = rate / 12 / 100
        n = years * 12
        emi = (amount * r * (1 + r) ** n) / ((1 + r) ** n - 1)
        st.success(f"Monthly EMI: ₹{emi:,.2f}")
        fig, ax = plt.subplots()
        ax.bar(["EMI"], [emi])
        ax.set_ylabel("Amount")
        st.pyplot(fig)

# ---------------------------
# Chatbot Assistant (interactive)
# ---------------------------
elif page == "Chatbot Assistant":
    st.title("🤖 Chatbot Assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    user_msg = st.text_input("Ask something about the project, prediction or loans")
    if st.button("Send"):
        if user_msg and user_msg.strip() != "":
            st.session_state.chat.append(("You", user_msg))
            msg = user_msg.lower()
            # rules-based answers
            if "price" in msg or "predict" in msg or "estimate" in msg:
                reply = "To predict, provide bedrooms, bathrooms, living area, condition (1-5), and schools nearby. Use the Document Summary page to auto-extract from listing text."
            elif "emi" in msg or "loan" in msg:
                reply = "Open EMI Calculator from the sidebar to compute monthly payments."
            elif "shap" in msg:
                reply = "SHAP explains feature effects. If shap is installed and model compatible, use the SHAP page."
            elif "csv" in msg:
                reply = "We removed bulk CSV upload; use Document Summary or manual inputs per house. If you need CSV re-added I can restore it."
            else:
                reply = "Nice question! I can summarize listings and auto-extract features to predict price. Ask me to 'summarize' or 'predict' as an example."
            st.session_state.chat.append(("Bot", reply))
    # show chat history
    for sender, text in st.session_state.chat:
        if sender == "You":
            st.markdown(f"<div class='chat-bubble-user'><b>{sender}:</b> {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'><b>{sender}:</b> {text}</div>", unsafe_allow_html=True)