import joblib
import re
import string
import streamlit as st
import os
import matplotlib.pyplot as plt

# -------------------------------
# Load Models
# -------------------------------
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
    "Extreme Gradient Boosting (XGBoost)": XGB,
    "Gradient Boosting": GBC,
}

# -------------------------------
# Text Preprocessing
# -------------------------------
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

# -------------------------------
# Sidebar Model Descriptions
# -------------------------------
def model_details():
    st.sidebar.markdown("## 🧠 Model Descriptions")
    model_choice = st.sidebar.selectbox("Learn about a model:", ["All"] + list(models.keys()))

    descriptions = {
        "Logistic Regression": (
            "### Logistic Regression\n"
            "- **Type:** Linear Model\n"
            "- **How it works:** Uses a weighted sum of input features (word frequencies) and applies a logistic function to estimate probabilities.\n"
            "- **Strengths:** Simple, fast, interpretable, works well on linearly separable data.\n"
            "- **Limitations:** Struggles with non-linear patterns.\n"
            "- **Use case:** Great baseline for text classification like fake news detection."
        ),
        "Extreme Gradient Boosting (XGBoost)": (
            "### Extreme Gradient Boosting (XGBoost)\n"
            "- **Type:** Advanced Ensemble Method\n"
            "- **How it works:** Builds multiple decision trees sequentially. Each new tree focuses on correcting errors of previous trees.\n"
            "- **Strengths:** Handles non-linear relationships, robust against overfitting, excellent accuracy.\n"
            "- **Special Features:** Regularization, missing value handling, parallel computing.\n"
            "- **Use case:** Common in Kaggle competitions and real-world high-performance tasks."
        ),
        "Gradient Boosting": (
            "### Gradient Boosting Classifier\n"
            "- **Type:** Ensemble Method\n"
            "- **How it works:** Similar to XGBoost but less optimized. Builds trees sequentially to minimize error using gradient descent.\n"
            "- **Strengths:** Captures complex patterns better than single models.\n"
            "- **Limitations:** Slower than XGBoost and prone to overfitting if not tuned.\n"
            "- **Use case:** Suitable for datasets where relationships between words are non-linear."
        )
    }

    if model_choice == "All":
        for name, desc in descriptions.items():
            st.sidebar.markdown(f"**{name}**:\n\n{desc}\n\n")
    else:
        st.sidebar.markdown(descriptions[model_choice])

# -------------------------------
# Confidence Chart
# -------------------------------
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

# -------------------------------
# Main Streamlit App
# -------------------------------
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

if __name__ == "__main__":
    run_streamlit_app()
