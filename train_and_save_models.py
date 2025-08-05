import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

data_fake = pd.read_csv("FAKE.csv")
data_true = pd.read_csv("REAL.csv")

data_fake["class"] = 0
data_true["class"] = 1

data_fake.drop(data_fake.tail(10).index, inplace=True)
data_true.drop(data_true.tail(10).index, inplace=True)

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge[['title', 'class']].sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['title'] = data['title'].apply(clean_text)

x = data['title']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)

LR = LogisticRegression()
DT = DecisionTreeClassifier()
GB = GradientBoostingClassifier()

LR.fit(xv_train, y_train)
DT.fit(xv_train, y_train)
GB.fit(xv_train, y_train)

joblib.dump(LR, r"C:\Users\aarya\Dropbox\PC\Downloads\Classify Headlines AI\logistic_model.pkl")
joblib.dump(DT, r'C:\Users\aarya\Dropbox\PC\Downloads\Classify Headlines AI\decision_tree_model.pkl')
joblib.dump(GB, r'C:\Users\aarya\Dropbox\PC\Downloads\Classify Headlines AI\gradient_boosting_model.pkl')
joblib.dump(vectorizer, r'C:\Users\aarya\Dropbox\PC\Downloads\Classify Headlines AI\vectorizer.pkl')

print("Models trained and saved successfully.")