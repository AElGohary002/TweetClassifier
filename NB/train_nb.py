import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

# Define a simple preprocessing function
def preprocess(text):
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Set file paths
train_path = os.path.join(os.getcwd(),'OLIDv1.0', 'olid-training-v1.0.tsv')
test_path = os.path.join(os.getcwd(),'OLIDv1.0', 'testset-levela.tsv')
gold_path = os.path.join(os.getcwd(),'OLIDv1.0', 'labels-levela.csv')

# Load data
train_data = pd.read_csv(train_path, sep='\t')
test_data = pd.read_csv(test_path, sep='\t')
gold_standard = pd.read_csv(gold_path, header=None)

# Preprocess tweets
X_train = train_data['tweet'].apply(preprocess)
y_train = train_data['subtask_a']  # Assuming labels like 'OFF' and 'NOT'
X_test = test_data['tweet'].apply(preprocess)
y_test = gold_standard[1]

# Feature extraction with CountVectorizer
vectorizer = CountVectorizer(max_features=10000)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Random Oversampling if desired:
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# X_train_bow, y_train = ros.fit_resample(X_train_bow, y_train)

# Train the Naive Bayes classifier
start_train_time = time.perf_counter()
clf = MultinomialNB(alpha=1)
clf.fit(X_train_bow, y_train)
y_pred = clf.predict(X_test_bow)
end_train_time = time.perf_counter()
total_train_time = end_train_time - start_train_time

macro_f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1 Score:", macro_f1, "Runtime:", total_train_time)

# Save the trained model and vectorizer
with open("NB/nb_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("NB/nb_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Naive Bayes model and vectorizer saved.")
