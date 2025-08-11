import joblib
import re
import string
import streamlit as st
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import Counter

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
PAC = load_model("passive_aggressive_model.pkl")
SVM = load_model("svm_model.pkl")
NB = load_model("naive_bayes_model.pkl")
VOTING = load_model("voting_model.pkl")

models = {
    "Logistic Regression": LR,
    "Gradient Boosting": GBC,
    "Extreme Gradient Boosting (XGBoost)": XGB,
    "Passive Aggressive Classifier": PAC,
    "Linear SVM": SVM,
    "Naive Bayes": NB,
    "Voting Classifier (Soft)": VOTING
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

# ----------------- Model Info -----------------
def model_details():
    st.sidebar.markdown("## 🧠 Model Descriptions")
    model_choice = st.sidebar.selectbox("Learn about a model:", ["All"] + list(models.keys()))

    descriptions = {
        "Logistic Regression": "- **Type:** Linear Model\n- **Strengths:** Simple, interpretable, fast.",
        "Gradient Boosting": "- **Type:** Ensemble Method\n- **Strengths:** Handles complex relationships.",
        "Extreme Gradient Boosting (XGBoost)": "- **Type:** Advanced Ensemble Method\n- **Strengths:** High accuracy.",
        "Passive Aggressive Classifier": "- **Type:** Online Learning\n- **Strengths:** Fast for large text data.",
        "Linear SVM": "- **Type:** Linear Classifier\n- **Strengths:** Excellent for high-dimensional text data.",
        "Naive Bayes": "- **Type:** Probabilistic Classifier\n- **Strengths:** Extremely fast and efficient.",
        "Voting Classifier (Soft)": "- **Type:** Ensemble Method\n- **Strengths:** Stable and accurate."
    }

    if model_choice == "All":
        for name, desc in descriptions.items():
            st.sidebar.markdown(f"**{name}**:\n\n{desc}\n\n")
    else:
        st.sidebar.markdown(descriptions[model_choice])

# ----------------- Visualisation -----------------
def dataset_visualisation():
    st.header("📊 Dataset & Visualisation")

    # Load dataset
    if not os.path.exists("FAKE.csv") or not os.path.exists("REAL.csv"):
        st.error("FAKE.csv and REAL.csv not found.")
        return

    fake_df = pd.read_csv("FAKE.csv").head(500)
    real_df = pd.read_csv("REAL.csv").head(500)
    fake_df["class"] = 0
    real_df["class"] = 1
    df = pd.concat([fake_df, real_df])

    # Class distribution
    st.subheader("Class Distribution")
    class_counts = df["class"].value_counts()
    labels = ["Fake News", "Real News"]
    fig, ax = plt.subplots()
    ax.bar(labels, class_counts, color=["red", "green"])
    plt.ylabel("Count")
    st.pyplot(fig)

    # Word Clouds
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)

    fake_text = " ".join(wordopt(str(t)) for t in fake_df["title"])
    real_text = " ".join(wordopt(str(t)) for t in real_df["title"])

    with col1:
        st.markdown("**Fake News Word Cloud**")
        wc_fake = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(fake_text)
        plt.imshow(wc_fake, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())

    with col2:
        st.markdown("**Real News Word Cloud**")
        wc_real = WordCloud(width=400, height=300, background_color="white", colormap="Greens").generate(real_text)
        plt.imshow(wc_real, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())

    # Top Words
    st.subheader("Top 20 Most Frequent Words in Each Class")
    def plot_top_words(text, title, color):
        words = text.split()
        common_words = Counter(words).most_common(20)
        labels, counts = zip(*common_words)
        fig, ax = plt.subplots()
        ax.barh(labels, counts, color=color)
        ax.invert_yaxis()
        plt.title(title)
        st.pyplot(fig)

    plot_top_words(fake_text, "Fake News", "red")
    plot_top_words(real_text, "Real News", "green")

# ----------------- Confidence Chart -----------------
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

# ----------------- Streamlit App -----------------
def run_streamlit_app():
    st.title("📰 News Sentry")
    model_details()

    # Dataset visualisation section in dropdown/expander
    with st.expander(" Show Dataset Visualisation"):
        dataset_visualisation()

    st.markdown("### Try an Example Headline:")
    example = st.selectbox(
        "Pick an example or type your own below:",
        ["", "Breaking: US declares war on Mars", 
         "Aliens spotted at the White House", 
         "NASA announces successful moon mission"]
    )

    headline = st.text_input("Enter your own headline:", value=example if example else "")

    if headline:
        with st.spinner("Analyzing..."):
            processed = wordopt(headline)
            new_xv = vectorizer.transform([processed])
            predictions, confidences = [], []

            st.subheader("📊 Model Predictions")
            for name, model in models.items():
                prediction = model.predict(new_xv)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(new_xv)[0]
                    confidence = round(max(proba) * 100, 2)
                else:
                    confidence = 50.0
                predictions.append(prediction)
                confidences.append((name, (prediction, confidence)))
                color = "green" if prediction == 1 else "red"
                st.markdown(f"**{name}:** :{color}[{output_label(prediction)}]")
                st.markdown(f"Confidence: `{confidence}%`")
                st.markdown(f"Explanation: {get_explanation(prediction)}")
                st.markdown("---")

            show_confidence_chart(confidences)
            real_conf = sum(conf for (_, (pred, conf)) in confidences if pred == 1)
            fake_conf = sum(conf for (_, (pred, conf)) in confidences if pred == 0)
            final_vote = 1 if real_conf >= fake_conf else 0
            st.subheader("✅ Final Verdict")
            color = "green" if final_vote == 1 else "red"
            st.markdown(f"<h3 style='color:{color};'>Prediction: {output_label(final_vote)}</h3>", unsafe_allow_html=True)
            st.info(f"Model Votes — Real: {predictions.count(1)}, Fake: {predictions.count(0)}")

if __name__ == "__main__":
    run_streamlit_app()