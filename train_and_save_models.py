import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data_fake = pd.read_csv("FAKE.csv")
data_true = pd.read_csv("REAL.csv")

# Add labels
data_fake["class"] = 0
data_true["class"] = 1

# Remove last 10 rows (for consistency with original code)
data_fake.drop(data_fake.tail(10).index, inplace=True)
data_true.drop(data_true.tail(10).index, inplace=True)

# Merge and shuffle
data = pd.concat([data_fake, data_true], axis=0)
data = data[['title', 'class']].sample(frac=1, random_state=42).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text.strip()

# Clean titles
data['title'] = data['title'].apply(clean_text)

# Split data
X = data['title']
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF Vectorizer with bigrams for better context
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
LR = LogisticRegression(max_iter=3000)
GB = GradientBoostingClassifier()
XGB = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train models
LR.fit(X_train_vec, y_train)
GB.fit(X_train_vec, y_train)
XGB.fit(X_train_vec, y_train)

# Evaluate
print("Model Accuracy:")
print(f"Logistic Regression: {accuracy_score(y_test, LR.predict(X_test_vec)):.4f}")
print(f"Gradient Boosting: {accuracy_score(y_test, GB.predict(X_test_vec)):.4f}")
print(f"XGBoost: {accuracy_score(y_test, XGB.predict(X_test_vec)):.4f}")

# Save models
joblib.dump(LR, "logistic_model.pkl")
joblib.dump(GB, "gradient_boosting_model.pkl")
joblib.dump(XGB, "xgboost_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Models trained and saved successfully!")
