from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

app = Flask(__name__)


# Load the CNN Artifacts

cnn_model = load_model("CNN/cnn_model.h5")

with open("CNN/tokenizer.pkl", "rb") as f:
    cnn_tokenizer = pickle.load(f)

with open("CNN/max_len.pkl", "rb") as f:
    cnn_max_len = pickle.load(f)


# Load the Naive Bayes Artifacts

with open("NB/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("NB/nb_vectorizer.pkl", "rb") as f:
    nb_vectorizer = pickle.load(f)

# Define the NB preprocessing function
def preprocess(text):
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)


# Define Routes for the Flask App

@app.route('/')
def index():
    # Render an HTML page with a form including a dropdown to choose the model
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    model_choice = request.form['model_choice']
    
    if model_choice == 'cnn':
        # Process tweet using CNN pipeline
        sequence = cnn_tokenizer.texts_to_sequences([tweet])
        padded_sequence = pad_sequences(sequence, maxlen=cnn_max_len)
        prediction = cnn_model.predict(padded_sequence)
        pred_label = "OFF" if prediction[0][0] > 0.5 else "NOT"
    elif model_choice == 'naive_bayes':
        # Process tweet using Naive Bayes pipeline
        processed_tweet = preprocess(tweet)
        tweet_vector = nb_vectorizer.transform([processed_tweet])
        prediction = nb_model.predict(tweet_vector)
        pred_label = prediction[0]
    else:
        pred_label = "Unknown model selection."
    
    return render_template('index.html', tweet=tweet, prediction=pred_label, model_choice=model_choice)

if __name__ == '__main__':
    app.run(debug=True)
