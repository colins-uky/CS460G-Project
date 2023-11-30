# NumPy and Pandas
import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf


# Keras - top level api for tensorflow
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, TextVectorization, LSTM, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import time
import re
import string

# import data helper functions
from data import get_data_iterator, get_entire_dataset_as_tfds





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
USECOLS = ["review", "recommended"]

# Number of distinct words in dataset
VOCAB_SIZE = 1000
 

# Number of rows to read per chunk
CHUNKSIZE = 500

BUFFER_SIZE = 10000
BATCH_SIZE = 64


EMBEDDING_DIM = 64
EPOCHS = 50
TEST_SIZE = 0.2



MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250


####### END GLOBAL VARS #######



####### PREPROCESSING #######

def filter_input(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')



# get dataset from .csv
train_dataset, test_dataset = get_entire_dataset_as_tfds(file_path=GPT_REVIEWS_PATH, 
                                                        use_cols=USECOLS, 
                                                        test_size=TEST_SIZE
                                                    )




  
# Initialize word Encoder
encoder = TextVectorization(
    standardize=filter_input,
    max_tokens=MAX_FEATURES,  
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
encoder.adapt(train_dataset.map(lambda text, label: text))


def encode_text(text, label):
    text = tf.expand_dims(text, -1)
    label = tf.expand_dims(label, -1)
    return encoder(text), label





train_dataset = train_dataset.map(encode_text)
test_dataset = test_dataset.map(encode_text)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)



briuh = 10
####### END PREPROCESSING #######



####### BUILD MODEL #######

class RNN:
    def __init__(self) -> None:
        
        
     
        self.loss = 0.0
        self.accuracy = 0.0
        
        
        
        self.model = Sequential([
            Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim = EMBEDDING_DIM,
                # Mask paddings with zeros
                mask_zero=True
            ),
            Bidirectional(LSTM(EMBEDDING_DIM)),
            Dense(EMBEDDING_DIM, activation='relu'),
            Dense(1, activation="sigmoid")
        ])  
        
        
        # Compile Model
        self.model.compile(
            loss=BinaryCrossentropy(from_logits=False),
            optimizer=Adam(0.001),
            metrics=['accuracy']
        )
        
        
        
    def train(self):
        self.model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, validation_steps=30)
        
        
    def test(self):
        self.loss, self.accuracy = self.model.evaluate(test_dataset)
        
        
    def print_score(self):
        print()
        print(f"Loss:       {self.loss}")
        print()
        print(f"Accuracy:   {self.accuracy}")
        

print()
print()
print()
print()
print()
print()
print()
print()
print()


def main():
    
    rnn = RNN()
    
    rnn.train()
    
    rnn.test()
    
    rnn.print_score()
    
    
    
        
    
    
    
if __name__ == "__main__":
    main()
    






