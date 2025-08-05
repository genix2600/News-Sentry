import joblib
import pandas as pd
import re
import string
import streamlit as st
import os
import matplotlib.pyplot as plt

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

vectorizer = load_model('vectorizer.pkl')
LR = load_model("logistic_model.pkl")
XGB = load_model("xgboost_model.pkl")
GBC = load_model("gradient_boosting_model.pkl")

models = {
    "Logistic Regression": LR,
    "Extreme Gradient Boosting": XGB,
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
    st.sidebar.markdown("## 🧠 Model Descriptions")
    model_choice = st.sidebar.selectbox("Learn about a model:", ["All"] + list(models.keys()))

    descriptions = {
        "Logistic Regression": (
            "### Logistic Regression\n"
            "- Works like a **smart calculator**\n"
            "- Counts how often words appear in a headline\n"
            "- Uses simple math to guess if it's real or fake\n"
            "- Fast, lightweight, and often surprisingly accurate\n"
            "- Like a **sharp memory** for word patterns"
        ),
        "Extreme Gradient Boosting": (
            "### Extreme Gradient Boosting (XGBoost)\n"
            "- Advanced boosting algorithm optimized for speed and accuracy\n"
            "- Handles complex patterns better than standard Gradient Boosting\n"
            "- Often used in **Kaggle competitions** because of its performance"
        ),
        "Gradient Boosting": (
            "### Gradient Boosting\n"
            "- Like a **team of students** solving a problem\n"
            "- Each model learns from the mistakes of the last\n"
            "- Combines all learnings into a smart final decision\n"
            "- Slower, but usually more accurate than single models"
        )
    }

    if model_choice == "All":
        for name, desc in descriptions.items():
            st.sidebar.markdown(f"**{name}**: {desc}")
    else:
        st.sidebar.markdown(f"**{model_choice}**: {descriptions[model_choice]}")

def show_confidence_chart(confidences):
    st.subheader("📈 Confidence Comparison")
    model_names = [name for name, _ in confidences]
    conf_values = [conf for _, (_, conf) in confidences]

    colors = ['#4CAF50' if pred == 1 else '#F44336' for _, (pred, _) in confidences]

    fig, ax = plt.subplots()
    ax.barh(model_names, conf_values, color=colors)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)

    for i, v in enumerate(conf_values):
        ax.text(v + 1, i, f"{v}%", va='center')

    st.pyplot(fig)

def run_streamlit_app():
    st.title("📰 Fake News Headline Classifier")
    st.markdown("Enter a news headline and let **three powerful ML models** analyze whether it's **Fake or Real**.")
    model_details()

    st.markdown("### Try an Example Headline:")
    example = st.selectbox(
        "Pick an example or type your own below:",
        [
            "",
            "Breaking: US declares war on Mars",
            "Aliens spotted at the White House",
            "Elon Musk buys the moon for $1 trillion",
            "World to end tomorrow, scientists confirm",
            "Man claims he traveled through time to stop pandemic",
            "NASA announces successful moon mission",
            "COVID-19 vaccines approved worldwide",
            "Apple unveils new iPhone with AI-powered features",
            "Stock markets hit record highs after tech surge",
            "WHO warns about new global health concerns"
        ]
    )

    headline = st.text_input("Or enter your own headline:", value=example if example else "")

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
                confidences.append((name, (prediction, confidence)))

                color = "green" if prediction == 1 else "red"
                st.markdown(f"**{name}:** :{color}[{label}]")
                st.markdown(f"Confidence: `{confidence}%`")
                st.markdown(f"Explanation: {get_explanation(prediction)}")
                st.markdown("---")

            show_confidence_chart(confidences)

            real_conf = sum(conf for (_, (pred, conf)) in confidences if pred == 1)
            fake_conf = sum(conf for (_, (pred, conf)) in confidences if pred == 0)
            final_vote = 1 if real_conf >= fake_conf else 0

            final_label = output_label(final_vote)
            real_count = predictions.count(1)
            fake_count = predictions.count(0)
            confidence_percent = round(max(real_conf, fake_conf) / len(models), 2)

            st.subheader("✅ Final Verdict")

            color = "green" if final_vote == 1 else "red"
            st.markdown(
                f"<h3 style='color:{color};'>Prediction: {final_label}</h3>",
                unsafe_allow_html=True
            )

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
        confidences.append((name, confidence))

        print(f"\n{name}")
        print(f"Prediction:  {label}")
        print(f"Confidence:  {confidence}%")
        print(f"Explanation: {explanation}")

    real_conf = sum(conf for (name, conf), pred in zip(confidences, predictions) if pred == 1)
    fake_conf = sum(conf for (name, conf), pred in zip(confidences, predictions) if pred == 0)
    final_vote = 1 if real_conf >= fake_conf else 0

    final_label = output_label(final_vote)
    real_count = predictions.count(1)
    fake_count = predictions.count(0)
    confidence_percent = round(max(real_conf, fake_conf) / len(models), 2)

    print("\nFinal Verdict")
    print(f"Prediction: {final_label}")
    print(f"Votes - Real: {real_count}, Fake: {fake_count}")
    print(f"Confidence: {confidence_percent}%")

if __name__ == "__main__":
    run_streamlit_app()
