import joblib
import pandas as pd
import re
import string
import streamlit as st
import os


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
    return "Probably not fake news" if n == 1 else "Probably fake news"

def get_explanation(prediction):
    return (
        "🟢 This headline resembles real news patterns."
        if prediction == 1 else
        "🔴 This headline shows common signs of fake news."
    )

def model_details():
    st.sidebar.markdown("## Model Descriptions")
    model_choice = st.sidebar.selectbox("Learn about a model:", ["All"] + list(models.keys()))

    descriptions = {
        "Logistic Regression": (
            "### Logistic Regression\n"
            "-  Works like a **smart calculator**\n"
            "-  Counts how often words appear in a headline\n"
            "-  Uses simple math to guess if it's real or fake\n"
            "-  Fast, lightweight, and often surprisingly accurate\n"
            "-  Like a **sharp memory** for word patterns"
        ),
        "Decision Tree": (
            "###  Decision Tree\n"
            "-  Asks **yes/no questions**, like:\n"
            "- 'Is the headline emotional?'\n"
            "- 'Does it contain a number?'\n"
            "-  Follows a decision **flowchart** based on the answers\n"
            "-  Simple, visual, and easy to understand\n"
            "-  Not always perfect — can be too confident with little info"
        ),
        "Gradient Boosting": (
            "###  Gradient Boosting\n"
            "-  Like a **team of students** solving a problem\n"
            "-  Each model learns from the mistakes of the last\n"
            "-  Combines all learnings into a smart final decision\n"
            "-  Slower, but usually more accurate than single models"
        )
    }


























    if model_choice == "All":
        for name, desc in descriptions.items():
            st.sidebar.markdown(f"**{name}**: {desc}")
    else:
        st.sidebar.markdown(f"**{model_choice}**: {descriptions[model_choice]}")


def run_streamlit_app():
    st.title("📰 Fake News Headline Classifier")
    model_details()

    headline = st.text_input("Enter a news headline:")




















































































































    if headline:
        with st.spinner("Analyzing..."):
            processed = wordopt(headline)
            new_xv = vectorizer.transform([processed])

            predictions = []
            confidences = []

            st.subheader("📊 Model Predictions")

            for name, model in models.items():
                prediction = model.predict(new_xv)[0]
                proba = model.predict_proba(new_xv)[0]
                label = output_label(prediction)
                confidence = round(max(proba) * 100, 2)

                predictions.append(prediction)
                confidences.append((prediction, confidence))

                color = "green" if prediction == 1 else "red"
                st.markdown(f"**{name}:** :{color}[{label}]")
                st.markdown(f"Confidence: `{confidence}%`")
                st.markdown(f"Explanation: {get_explanation(prediction)}")
                st.markdown("---")

            final_vote = max(set(predictions), key=predictions.count)
            final_label = output_label(final_vote)
            real_count = predictions.count(1)
            fake_count = predictions.count(0)
            matching_conf = [conf for pred, conf in confidences if pred == final_vote]
            confidence_percent = round(sum(matching_conf) / len(matching_conf), 2) if matching_conf else 0.0

            st.subheader("Final Verdict")
            st.success(f"Prediction: {final_label}")
            st.info(f"Model Votes — Real: {real_count}, Fake: {fake_count}")
            st.info(f"Overall Confidence: {confidence_percent}%")

def manual_testing(news):
    processed = wordopt(news)
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

        print(f"\n{name}")
        print(f"Prediction:  {label}")
        print(f"Confidence:  {confidence}%")
        print(f"Explanation: {explanation}")

    final_vote = max(set(predictions), key=predictions.count)
    final_label = output_label(final_vote)
    real_count = predictions.count(1)
    fake_count = predictions.count(0)
    matching_conf = [conf for pred, conf in confidences if pred == final_vote]
    confidence_percent = round(sum(matching_conf) / len(matching_conf), 2) if matching_conf else 0.0

    print("\nFinal Verdict")
    print(f"Prediction: {final_label}")
    print(f"Votes - Real: {real_count}, Fake: {fake_count}")
    print(f"Confidence: {confidence_percent}%")


if __name__ == "__main__":
    run_streamlit_app()
