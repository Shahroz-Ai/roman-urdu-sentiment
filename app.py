import re

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Roman Urdu Sentiment", page_icon="💬")
st.title("💬 Roman Urdu Sentiment Analyzer")
st.caption("Type a sentence in Roman Urdu and see if it sounds positive, neutral, or negative.")


def clean_text(text):
    """Must stay identical to the cleaning used in prepare_data.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()
classes = list(model.named_steps["clf"].classes_)
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

EXAMPLES = [
    "yeh phone bohat acha hai",
    "bakwaas service, paisa barbad",
    "kal meeting subah 10 baje hai",
    "wah kya baat hai, kamal kar diya",
]


def set_text(t):
    st.session_state.text = t


st.write("Try an example:")
for ex in EXAMPLES:
    st.button(ex, on_click=set_text, args=(ex,))

text = st.text_area("Your sentence", key="text", height=100)

if st.button("Analyze", type="primary"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Please type some text first.")
    else:
        proba = model.predict_proba([cleaned])[0]
        top = classes[int(proba.argmax())]
        st.subheader(f"{EMOJI[top]} {top}")
        st.bar_chart(pd.DataFrame({"Probability": proba}, index=classes))
        if proba.max() < 0.5:
            st.info("The model is not sure about this one. The probabilities are close.")

with st.expander("About this model"):
    st.markdown(
        """
- Trained on about 19,000 Roman Urdu texts (UCI Roman Urdu Data Set, Zareen Sharf)
- Word and character TF-IDF features with Logistic Regression
- Macro F1 of about 0.67 on a held-out test set
- Labels in the dataset are noisy, and the model does not understand context (for example, who won a match)
- Probabilities show how the model leans, not how likely it is to be correct
"""
    )