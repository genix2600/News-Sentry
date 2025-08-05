import joblib
import pandas as pd
import re
import string
import streamlit as st
import os

# Load models and vectorizer
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

vectorizer = load_model('vectorizer.pkl')
LR = load_model("logistic_model.pkl")
DT = load_model("decision_tree_model.pkl")
GBC = load_model("gradient_boosting_model.pkl")

models = {
    "Logistic Regression": LR,
    "Decision Tree": DT,
    "Gradient Boosting": GBC,
}

# Text cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def output_label(n):
    return "🟢 Probably *not* fake news" if n == 1 else "🔴 Probably *fake* news"

def get_explanation(prediction):
    return (
        "✅ This headline resembles **real** news patterns."
        if prediction == 1 else
        "⚠️ This headline shows common **fake news** signals."
    )

# Custom chat-style CSS
def inject_chat_style():
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem 1rem;
        max-width: 800px;
        margin: auto;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 100%;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #F1F0F0;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Main UI
def run_chat_ui():
    st.set_page_config(page_title="Fake News Classifier", layout="centered")
    inject_chat_style()

    st.markdown("<h1 style='text-align: center;'>📰 Fake News Classifier Chat</h1>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    headline = st.text_input("💬 You:", placeholder="Enter a news headline here...")

    if headline:
        st.markdown(f"<div class='chat-bubble user-msg'>{headline}</div>", unsafe_allow_html=True)

        processed = wordopt(headline)
        new_xv = vectorizer.transform([processed])

        predictions = []
        confidences = []

        for name, model in models.items():
            prediction = model.predict(new_xv)[0]
            proba = model.predict_proba(new_xv)[0]
            label = output_label(prediction)
            confidence = round(max(proba) * 100, 2)
            explanation = get_explanation(prediction)

            predictions.append(prediction)
            confidences.append((prediction, confidence))

            model_response = f"""
            **{name}**
            - Prediction: {label}  
            - Confidence: `{confidence}%`  
            - {explanation}
            """
            st.markdown(f"<div class='chat-bubble bot-msg'>{model_response}</div>", unsafe_allow_html=True)

        final_vote = max(set(predictions), key=predictions.count)
        final_label = output_label(final_vote)
        real_count = predictions.count(1)
        fake_count = predictions.count(0)
        matching_conf = [conf for pred, conf in confidences if pred == final_vote]
        confidence_percent = round(sum(matching_conf) / len(matching_conf), 2) if matching_conf else 0.0

        final_response = f"""
        🧠 **Final Verdict**
        - Overall Prediction: **{final_label}**
        - Model Votes: ✅ Real - {real_count} | ❌ Fake - {fake_count}  
        - Average Confidence: `{confidence_percent}%`
        """
        st.markdown(f"<div class='chat-bubble bot-msg'>{final_response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    run_chat_ui()
import joblib
import pandas as pd
import re
import string
import streamlit as st
import os

# Load models and vectorizer
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

vectorizer = load_model('vectorizer.pkl')
LR = load_model("logistic_model.pkl")
DT = load_model("decision_tree_model.pkl")
GBC = load_model("gradient_boosting_model.pkl")

models = {
    "Logistic Regression": LR,
    "Decision Tree": DT,
    "Gradient Boosting": GBC,
}

# Text cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def output_label(n):
    return "🟢 Probably *not* fake news" if n == 1 else "🔴 Probably *fake* news"

def get_explanation(prediction):
    return (
        "✅ This headline resembles **real** news patterns."
        if prediction == 1 else
        "⚠️ This headline shows common **fake news** signals."
    )

# Custom chat-style CSS
def inject_chat_style():
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem 1rem;
        max-width: 800px;
        margin: auto;
    }
    .chat-bubble {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 100%;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #F1F0F0;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Main UI
def run_chat_ui():
    st.set_page_config(page_title="Fake News Classifier", layout="centered")
    inject_chat_style()

    st.markdown("<h1 style='text-align: center;'>📰 Fake News Classifier Chat</h1>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    headline = st.text_input("💬 You:", placeholder="Enter a news headline here...")

    if headline:
        st.markdown(f"<div class='chat-bubble user-msg'>{headline}</div>", unsafe_allow_html=True)

        processed = wordopt(headline)
        new_xv = vectorizer.transform([processed])

        predictions = []
        confidences = []

        for name, model in models.items():
            prediction = model.predict(new_xv)[0]
            proba = model.predict_proba(new_xv)[0]
            label = output_label(prediction)
            confidence = round(max(proba) * 100, 2)
            explanation = get_explanation(prediction)

            predictions.append(prediction)
            confidences.append((prediction, confidence))

            model_response = f"""
            **{name}**
            - Prediction: {label}  
            - Confidence: `{confidence}%`  
            - {explanation}
            """
            st.markdown(f"<div class='chat-bubble bot-msg'>{model_response}</div>", unsafe_allow_html=True)

        final_vote = max(set(predictions), key=predictions.count)
        final_label = output_label(final_vote)
        real_count = predictions.count(1)
        fake_count = predictions.count(0)
        matching_conf = [conf for pred, conf in confidences if pred == final_vote]
        confidence_percent = round(sum(matching_conf) / len(matching_conf), 2) if matching_conf else 0.0

        final_response = f"""
        🧠 **Final Verdict**
        - Overall Prediction: **{final_label}**
        - Model Votes: ✅ Real - {real_count} | ❌ Fake - {fake_count}  
        - Average Confidence: `{confidence_percent}%`
        """
        st.markdown(f"<div class='chat-bubble bot-msg'>{final_response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    run_chat_ui()
