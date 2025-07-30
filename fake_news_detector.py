import joblib
import pandas as pd
import re
import string

vectorizer = joblib.load("vectorizer.pkl")
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boosting_model.pkl")
RF = joblib.load("random_forest_model.pkl")

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Probably not fake news" if n == 1 else "Probably fake news"

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

        explanation = (
            "This headline seems factual and resembles real news patterns."
            if prediction == 1 else
            "This headline shows signs commonly seen in fake news — possibly emotional or misleading phrasing."
        )

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

    print("\n Final Verdict")
    print(f"Prediction: {final_label}")
    print(f"Based on {real_count} real and {fake_count} fake votes out of {len(predictions)} models.")
    print(f"Overall Confidence (weighted): {confidence_percent}%")


def model_info():
    print("\nAvailable Models and Their Roles:")
    print("- Logistic Regression (LR): Predicts the probability of a headline being real or fake using simple word patterns.")
    print("- Decision Tree (DT): Breaks down headline text features into a tree of decisions to classify them.")
    print("- Gradient Boosting (GBC): Builds multiple improving trees to detect subtle patterns of real vs fake news.")
    print("- Random Forest (RF): Uses many decision trees and takes a vote to decide if the headline is real or fake.\n")

    model_choice = input("Type the model name (e.g., 'LR', 'DT', 'GBC', 'RF') for more details, or 'back' to go back: ").strip().upper()

    if model_choice == "LR":
        print("\nLogistic Regression (LR):")
        print(" - This model checks which words commonly appear in real or fake headlines.")
        print(" - It assigns weights to these words and calculates a probability score.")
        print(" - For example, clickbait words like 'shocking' or 'you won't believe' might lower the realness score.")
        print(" - Fast and works well when there's a clear difference in word usage.")
    elif model_choice == "DT":
        print("\nDecision Tree (DT):")
        print(" - Breaks headlines into conditions like 'Does this word appear?' or 'Is this phrase used?'.")
        print(" - Follows these rules step-by-step like a flowchart to make a decision.")
        print(" - Easy to understand and explain, but can sometimes overfit on small patterns.")
    elif model_choice == "GBC":
        print("\nGradient Boosting Classifier (GBC):")
        print(" - This model builds a series of decision trees.")
        print(" - Each new tree corrects the mistakes made by the previous one.")
        print(" - It's excellent at picking up complex combinations of words or patterns that single trees might miss.")
        print(" - Takes longer to train but usually gives better accuracy.")
    elif model_choice == "RF":
        print("\nRandom Forest (RF):")
        print(" - Builds many decision trees on different parts of the data.")
        print(" - Each tree votes on whether the headline is real or fake.")
        print(" - The majority vote becomes the final prediction.")
        print(" - This makes it very stable and less prone to errors from individual trees.")
    elif model_choice == "BACK":
        return
    else:
        print("Invalid input. Type one of: LR, DT, GBC, RF or 'back' to return.")

while True:
    news = input("\nEnter headline (or type 'help' for model info, 'exit' to quit): ").strip()
    
    if news.lower() == "exit":
        print("Exiting.")
        break
    elif news.lower() == "help":
        model_info()
    else:
        manual_testing(news)