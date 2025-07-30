import joblib
import pandas as pd
import re
import string
import streamlit as st
import sys

vectorizer = joblib.load("vectorizer.pkl")
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boosting_model.pkl")
RF = joblib.load("random_forest_model.pkl")

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Probably not fake news" if n == 1 else "Probably fake news"

def get_explanation(prediction):
    return (
        "This headline seems factual and resembles real news patterns."
        if prediction == 1 else
        "This headline shows signs commonly seen in fake news — possibly emotional or misleading phrasing."
    )

def model_details():
    st.sidebar.markdown("### Model Descriptions")
    st.sidebar.markdown("**Logistic Regression (LR):** Simple model using word frequency patterns.")
    st.sidebar.markdown("**Decision Tree (DT):** Rule-based system analyzing text features.")
    st.sidebar.markdown("**Gradient Boosting (GBC):** Builds multiple trees, learns from mistakes.")
    st.sidebar.markdown("**Random Forest (RF):** Collection of decision trees voting together.")

def run_streamlit_app():
    st.title("Fake News Classifier")
    model_details()

    headline = st.text_input("Enter a news headline:")

    if headline:
        processed = wordopt(headline)
        new_xv = vectorizer.transform([processed])

        models = {
            "Logistic Regression": LR,
            "Decision Tree": DT,
            "Gradient Boosting": GB,
            "Random Forest": RF
        }

        st.subheader("Model Predictions")
        predictions = []
        confidences = []

        for name, model in models.items():
            prediction = model.predict(new_xv)[0]
            proba = model.predict_proba(new_xv)[0]
            label = output_label(prediction)
            confidence = round(max(proba) * 100, 2)
            predictions.append(prediction)
            confidences.append((prediction, confidence))

            color = "green" if prediction == 1 else "red"
            st.markdown(f"**{name}:** :{color}[{label}]  ")
            st.markdown(f"Confidence: `{confidence}%`  ")
            st.markdown(f"Explanation: {get_explanation(prediction)}")
            st.markdown("---")

        final_vote = max(set(predictions), key=predictions.count)
        final_label = output_label(final_vote)
        real_count = predictions.count(1)
        fake_count = predictions.count(0)
        matching_conf = [conf for pred, conf in confidences if pred == final_vote]
        confidence_percent = round(sum(matching_conf) / len(matching_conf), 2)

        st.subheader("🧾 Final Verdict")
        st.success(f"Prediction: {final_label}")
        st.info(f"Models voted — Real: {real_count}, Fake: {fake_count}")
        st.info(f"Overall Confidence: {confidence_percent}%")

def manual_testing(news):
    new_data = pd.DataFrame({'title': [news]})
    new_data['title'] = new_data['title'].apply(wordopt)
    new_xv = vectorizer.transform(new_data['title'])

    models = {
        "Logistic Regression": LR,
        "Decision Tree": DT,
        "Gradient Boosting": GB,
        "Random Forest": RF
    }

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
    matching_confidences = [conf for pred, conf in confidences if pred == final_vote]
    confidence_percent = round(sum(matching_confidences) / len(matching_confidences), 2)

    print("\nFinal Verdict")
    print(f"Prediction: {final_label}")
    print(f"Based on {real_count} real and {fake_count} fake votes out of {len(predictions)} models.")
    print(f"Overall Confidence (weighted): {confidence_percent}%")

def model_info():
    print("\nAvailable Models and Their Roles:")
    print("- Logistic Regression (LR): Predicts using word frequency patterns.")
    print("- Decision Tree (DT): Rule-based breakdown of features.")
    print("- Gradient Boosting (GBC): Learns from previous mistakes across multiple trees.")
    print("- Random Forest (RF): Uses multiple trees to vote on the result.\n")

    model_choice = input("Type the model name (LR, DT, GBC, RF) or 'back': ").strip().upper()
    if model_choice == "LR":
        print("LR: Weights word patterns and predicts probability.")
    elif model_choice == "DT":
        print("DT: Follows yes/no paths to classify.")
    elif model_choice == "GBC":
        print("GBC: Builds on errors to improve.")
    elif model_choice == "RF":
        print("RF: Ensemble of trees that vote.")
    elif model_choice == "BACK":
        return
    else:
        print("Invalid input.")

if __name__ == "__main__":
    if hasattr(st, "_is_running_with_streamlit") and st._is_running_with_streamlit:
        run_streamlit_app()
    else:
        while True:
            news = input("\nEnter headline (or type 'help' for model info, 'exit' to quit): ").strip()
            if news.lower() == "exit":
                print("Exiting.")
                break
            elif news.lower() == "help":
                model_info()
            else:
                manual_testing(news)