import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
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

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
xv_train = vectorizer.fit_transform(x_train)

LR = LogisticRegression(max_iter=3000)
GB = GradientBoostingClassifier()
XGB = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
PAC = PassiveAggressiveClassifier(max_iter=2000)
SVM = LinearSVC()
NB = MultinomialNB()

LR.fit(xv_train, y_train)
GB.fit(xv_train, y_train)
XGB.fit(xv_train, y_train)
PAC.fit(xv_train, y_train)
SVM.fit(xv_train, y_train)
NB.fit(xv_train, y_train)

voting_models = [
    ('lr', LR),
    ('gb', GB),
    ('xgb', XGB),
    ('pac', PAC),
    ('svm', SVM),
    ('nb', NB)
]

voting_soft = VotingClassifier(estimators=[
    ('lr', LR),
    ('gb', GB),
    ('xgb', XGB),
    ('nb', NB)
], voting='soft')

voting_soft.fit(xv_train, y_train)

joblib.dump(LR, 'logistic_model.pkl')
joblib.dump(GB, 'gradient_boosting_model.pkl')
joblib.dump(XGB, 'xgboost_model.pkl')
joblib.dump(PAC, 'passive_aggressive_model.pkl')
joblib.dump(SVM, 'svm_model.pkl')
joblib.dump(NB, 'naive_bayes_model.pkl')
joblib.dump(voting_soft, 'voting_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ All models trained and saved successfully!")