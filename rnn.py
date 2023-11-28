# NumPy and Pandas
import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf


# Keras - top level api for tensorflow
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, TextVectorization, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# import data helper functions
from data import get_data_iterator





####### INITIALIZE GLOBAL VARS #######

# (7.98 GB FILE WARNING) 
# Use get_data_generator to load data in chunks into memory
STEAM_REVIEWS_FILEPATH = "data/archive/steam_reviews.csv"

# Smaller dataset for testing (30 rows)
GPT_REVIEWS_PATH = "data/test/gpt_reviews.csv"

# Name of columns to read
#review: text of review, recommended: boolean
USECOLS = ["review", "recommended"]

# Number of distinct words in dataset
VOCAB_SIZE = 1000


# Number of rows to read per chunk
CHUNKSIZE = 10

BUFFER_SIZE = 10000
BATCH_SIZE = 64

EMBEDDING_DIM = 64
EPOCHS = 250
TEST_SIZE = 0.2


####### END GLOBAL VARS #######



####### PREPROCESSING #######

data_gen = get_data_iterator(GPT_REVIEWS_PATH, CHUNKSIZE, USECOLS)

# Initialize word Tokenizer
tokenizer = Tokenizer()

# Initialize word Encoder
encoder = TextVectorization(max_tokens=VOCAB_SIZE)


####### END PREPROCESSING #######



####### BUILD MODEL #######

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim = EMBEDDING_DIM,
        # Mask paddings with zeros
        mask_zero=True
    ),
    Bidirectional(LSTM(EMBEDDING_DIM)),
    Dense(EMBEDDING_DIM, activation='relu'),
    Dense(1)
])



# Compile Model
model.compile(loss=BinaryCrossentropy(from_logits=True),
              optimizer=Adam(1e-4),
              metrics=['accuracy']
            )



# Helper function to encode a chunk of data
def preprocess_chunk(chunk, tokenizer, encoder):
    
    
    
    texts = tf.make_ndarray(chunk[0]['review']).tolist()
    labels = chunk[1]
    

    
    tokenizer.fit_on_texts(texts)
    encoder.adapt(texts)
    encoded_texts = encoder(texts)
    
    return encoded_texts, labels


# Train the model

for epoch in range(EPOCHS):
    for chunk in data_gen.take(1):
        
        encoded_texts, labels = preprocess_chunk(chunk, tokenizer, encoder)
        model.train_on_batch(encoded_texts, labels)




test_gen = get_data_iterator(GPT_REVIEWS_PATH, CHUNKSIZE, USECOLS)
# Evaluation on the test set
total_loss = 0.0
total_accuracy = 0.0
total_batches = 0

for chunk in test_gen:
    encoded_texts, labels = preprocess_chunk(chunk, tokenizer, encoder)
    loss, accuracy = model.evaluate(encoded_texts, labels, verbose=0)
    total_loss += loss
    total_accuracy += accuracy
    total_batches += 1

average_loss = total_loss / total_batches
average_accuracy = total_accuracy / total_batches

print(f"\n\nAverage Loss: {average_loss}.\nAverage Accuracy: {average_accuracy}.\n\n")