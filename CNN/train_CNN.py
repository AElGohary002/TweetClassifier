import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam

# Data Loading and Preprocessing
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
train_path = os.path.join(os.getcwd(),'OLIDv1.0', 'olid-training-v1.0.tsv')
test_path = os.path.join(os.getcwd(),'OLIDv1.0', 'testset-levela.tsv')
gold_path = os.path.join(os.getcwd(),'OLIDv1.0', 'labels-levela.csv')

train_df = pd.read_csv(train_path, delimiter='\t', header=0, names=['id', 'tweet', 'label', 'sub_a', 'sub_b'])
test_df = pd.read_csv(test_path, delimiter='\t', header=0, names=['id', 'tweet'])
gold_df = pd.read_csv(gold_path, header=None, names=['id', 'label'])

test_df = test_df.merge(gold_df, on='id')

# Encode labels
label_map = {'OFF': 0, 'NOT': 1}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Tokenisation and Padding
max_features = 12000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_df['tweet'])
train_sequences = tokenizer.texts_to_sequences(train_df['tweet'])
test_sequences = tokenizer.texts_to_sequences(test_df['tweet'])

# Determine maximum sequence length from training data
max_len = max([len(s) for s in train_sequences])
train_sequences = pad_sequences(train_sequences, maxlen=max_len)
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

# Build CNN
embedding_dim = 100
num_filters = 128
kernel_size = 3

model = Sequential([
    Embedding(max_features, embedding_dim, input_length=max_len),
    Conv1D(num_filters, kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train CNN
batch_size = 32
epochs = 2
model.fit(train_sequences, train_df['label'], batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

# Evaluation
test_predictions = model.predict(test_sequences)
test_predictions = np.round(test_predictions).astype(int).flatten()
macro_f1 = f1_score(test_df['label'], test_predictions, average='macro')
print("Macro F1 Score: ", macro_f1)

# Save model and tokeniser
model.save("CNN/cnn_model.h5")  # Save the trained model

with open("CNN/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save the maximum sequence length for later use in padding
with open("CNN/max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

print("Model, tokenizer, and max_len saved.")
