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

import time

# import data helper functions
from data import get_data_iterator





####### INITIALIZE GLOBAL VARS #######

# (7.98 GB FILE WARNING) 
# Use get_data_iterator to load data in chunks into memory
STEAM_REVIEWS_LARGE_PATH = "data/archive/steam_reviews_large.csv"

# (2.01 GB FILE WARNING)
STEAM_REVIEWS_SMALL_PATH = "data/archive/steam_reviews_small.csv"

# Smaller dataset for testing (30 rows)
GPT_REVIEWS_PATH = "data/test/gpt_reviews.csv"

# Name of columns to read
#review: text of review, recommended: boolean
USECOLS = ["review_text", "review_score"]

# Number of distinct words in dataset
VOCAB_SIZE = 1000


# Number of rows to read per chunk
CHUNKSIZE = 500

BUFFER_SIZE = 10000
BATCH_SIZE = 64

EMBEDDING_DIM = 64
EPOCHS = 100
TEST_SIZE = 0.2


####### END GLOBAL VARS #######



####### PREPROCESSING #######

data_gen = get_data_iterator(STEAM_REVIEWS_SMALL_PATH, CHUNKSIZE, USECOLS)




####### END PREPROCESSING #######



####### BUILD MODEL #######

class RNN:
    def __init__(self) -> None:
        self.model = Sequential([
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
        self.model.compile(loss=BinaryCrossentropy(from_logits=True),
              optimizer=Adam(0.01),
              metrics=['accuracy']
        )
        
        # Initialize word Tokenizer
        self.tokenizer = Tokenizer()

        # Initialize word Encoder
        self.encoder = TextVectorization(max_tokens=VOCAB_SIZE)
        
        
    def train(self):
        # Train the model
        start = time.time()
        for epoch in range(EPOCHS):
            print(f"epoch: {epoch + 1}")
            for chunk in data_gen:

                encoded_texts, labels = self.preprocess_chunk(chunk)
                self.model.train_on_batch(encoded_texts, labels)
    
        interval = time.time() - start

        print(f"Training finished. Took {interval:.2f} seconds.")


    # Helper function to encode a chunk of data
    def preprocess_chunk(self, chunk):
        labels = chunk[1]
        texts = [item.decode('utf-8') for item in chunk[0][USECOLS[0]].numpy()]
        
        self.tokenizer.fit_on_texts(texts)

        self.encoder.adapt(texts)
        encoded_texts = self.encoder(texts)
        
        return encoded_texts, labels


    def test(self):
        test_gen = get_data_iterator(STEAM_REVIEWS_SMALL_PATH, CHUNKSIZE, USECOLS)
        # Evaluation on the test set
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0


        print("Starting testing.")
        start = time.time()
        for chunk in test_gen:
            encoded_texts, labels = self.preprocess_chunk(chunk)
            loss, accuracy = self.model.evaluate(encoded_texts, labels, verbose=0)
            total_loss += loss
            total_accuracy += accuracy
            total_batches += 1
            print(f"Batches tested: {total_batches}.")
            if total_batches > 10:
                break
            
        interval = time.time() - start
        print(f"Finished testing. Took {interval:.3f} seconds.")







        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches

        print(f"\n\nAverage Loss: {average_loss}.\nAverage Accuracy: {average_accuracy}.\n\n")



    def predict(self, text):
        # Tokenize and encode the input text
        text = [text]
        self.encoder.adapt(text)
        encoded_text = self.encoder(text)

        # Make predictions
        predictions = self.model.predict(encoded_text)

        # The model output is logits. Apply sigmoid activation for binary classification.
        predicted_prob = tf.nn.sigmoid(predictions[0]).numpy()

        # Determine the sentiment based on the probability
        if predicted_prob >= 0.5:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        return sentiment, predicted_prob






def main():
    
    
    
    rnn = RNN()
    rnn.train()
    rnn.test()
    
    
    inp = input(">>")
    while inp != "q":
        sentiment, prob = rnn.predict(inp)
        
        if prob < 0.5:
            print(f"RNN predicts {sentiment} sentiment with {1 - prob} probability.")
        else:
            print(f"RNN predicts {sentiment} sentiment with {prob} probability.")
            
        inp = input(">>")
        
    
    
    
if __name__ == "__main__":
    main()
    






