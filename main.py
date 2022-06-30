"""
A machine learning model predicting emotional sentiment,
using the "Emotions dataset for NLP" dataset from Kaggle.

Zain Fox-Latif
@zfla
""" 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics     
from sklearn.pipeline import Pipeline

# Preparing data for usage
df_train = pd.read_csv("train.txt", delimiter=";", names=['text', 'label'])
df_test = pd.read_csv("test.txt", delimiter=";", names=['text', 'label'])

X_train = df_train["text"]
X_test = df_test["text"]

y_train = df_train["label"]
y_test = df_test["label"]

"""
Testing params

text_clf = Pipeline([
    ("vect", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("clf", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=12, max_iter=5, tol=None)),
])

params = {
    "vect__ngram_range" : ([1,3], [1,4], [1,5]),
    "tfidf__use_idf" : (True, False),
    "clf__penalty" : ("l2", "l1"),
    "clf__alpha" : (1e-4, 1e-5, 1e-6),
}

gs_clf = GridSearchCV(text_clf, params, cv=5, n_jobs=-1).fit(X_train, y_train)

y_pred = gs_clf.predict(X_test)

for name in params.keys():
    print("{} : {}".format(name, gs_clf.best_params_[name]))

print(gs_clf.best_score_)
"""

# Building a pipeline
text_clf = Pipeline([
    ("vect", CountVectorizer(ngram_range=[1,5])),
    ("tfidf", TfidfTransformer(use_idf=True)),
    ("clf", SGDClassifier(penalty="l1", alpha=1e-5, random_state=12, max_iter=5, tol=None)),
])

# Fitting/predicting data
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)

# Results
print("accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))