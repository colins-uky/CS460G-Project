# NumPy and Pandas
import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


# Keras - top level api for tensorflow
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, Embedding, TextVectorization, LSTM, Bidirectional, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import time
import re
import os
import string


gpu = tf.config.list_physical_devices('GPU')

if len(gpu) >= 1:
    print("GPU DETECTED: ", gpu)
else:
    print("No GPU detected, using cpu.")




####### INITIALIZE GLOBAL VARS #######


CWD = os.getcwd()


# (7.98 GB FILE WARNING) 
# Use get_data_iterator to load data in chunks into memory
STEAM_REVIEWS_LARGE_PATH = CWD + "/data/archive/steam_reviews_large.csv"

# (2.01 GB FILE WARNING)
STEAM_REVIEWS_SMALL_PATH = CWD + "/data/archive/steam_reviews_small.csv"

# Smaller dataset for testing (30 rows)
GPT_REVIEWS_PATH = CWD + "/data/test/gpt_reviews.csv"

# Name of columns to read
#review: text of review, recommended: boolean
USECOLS = ["review_text", "review_score"]

# Number of distinct words in dataset
VOCAB_SIZE = 1000
 
# Number of rows to read per chunk
CHUNKSIZE = 20

BUFFER_SIZE = 10000
BATCH_SIZE = 64


EMBEDDING_DIM = 64
EPOCHS = 3
TEST_SIZE = 0.2



MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250


####### END GLOBAL VARS #######



####### PREPROCESSING #######

def get_data_pipeline(filpath, chunksize, use_cols):
    it = tf.data.experimental.make_csv_dataset(
        file_pattern=filpath,
        batch_size=chunksize,
        select_columns=use_cols,
        label_name=use_cols[-1],
        num_epochs=1,
        shuffle=True
    )
    
    return it

def filter_input(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')



train_pipe = get_data_pipeline(
    filpath=STEAM_REVIEWS_SMALL_PATH,
    chunksize=CHUNKSIZE,
    use_cols=USECOLS
)

test_pipe = get_data_pipeline(
    filpath=STEAM_REVIEWS_SMALL_PATH,
    chunksize=CHUNKSIZE,
    use_cols=USECOLS
)

val_pipe = get_data_pipeline(
    filpath=STEAM_REVIEWS_SMALL_PATH,
    chunksize=CHUNKSIZE,
    use_cols=USECOLS
)

  
# Initialize word Encoder
encoder = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")


def encode_text(text, label):
    text = encoder(text["review_text"])
    text = tf.expand_dims(text, -1)
    label = tf.expand_dims(label, -1)
    
    return text, label



AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_pipe.map(encode_text)
val_dataset = val_pipe.map(encode_text)
test_dataset = test_pipe.map(encode_text)


train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)






briuh = 10
####### END PREPROCESSING #######



####### BUILD MODEL #######

class RNN:
    def __init__(self) -> None:
        
        
     
        self.loss = 0.0
        self.accuracy = 0.0
        
        
        self.model = None
        
        
    
    def build(self):
        self.model = Sequential([
            LSTM(units=EMBEDDING_DIM, return_sequences=True),
            Dropout(0.2),
            LSTM(units=EMBEDDING_DIM, return_sequences=False),
            Dropout(0.2),
            Dense(EMBEDDING_DIM, activation='relu'),
            Dense(1, activation="sigmoid")
        ])
        
        # Compile Model
        self.model.compile(
            loss=BinaryCrossentropy(from_logits=False),
            optimizer=Adam(0.001),
            metrics=['accuracy']
        )
        
        
        
        
    
        
    def train(self, steps_per_epoch=None):
        
        self.model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, validation_steps=30, steps_per_epoch=steps_per_epoch)
        
        
    def test(self):
        self.loss, self.accuracy = self.model.evaluate(test_dataset, steps=10)
        
        
    def print_score(self):
        print()
        print(f"Loss:       {self.loss}")
        print()
        print(f"Accuracy:   {self.accuracy}")
        
        
        
    def predict(self, inp: str):
        
        text = encoder([inp])
        text = tf.expand_dims(text, -1)
        
        pred = self.model.predict(text)
        
        return pred
        
        
        
    
        
    def save_to_disk(self):
        cwd = os.path.join(CWD)
        filepath = os.path.join(cwd, "bin", "rnn.keras")

        answer = input(f"\nSave model to {filepath} ? \n(y/n) >> ")
        
        if answer.lower() == 'y':
            self.model.save(filepath)
        else:
            print("Canceled save.")
        
    
    def load_from_disk(self):
        cwd = os.path.join(CWD)
        filepath = os.path.join(cwd, "bin", "rnn.keras")

        answer = input(f"\nLoad model from {filepath} ? \n(y/n) >> ")
        
        if answer.lower() == 'y':
            self.model = load_model(filepath)
        else:
            print("Canceled load.")

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
    
    rnn.build()
    
    rnn.train(steps_per_epoch=20)
    
    rnn.test()
    
    rnn.print_score()
    
    rnn.save_to_disk()
    
    
    inp = input(">> ")
    while inp != 'q':
        pred = rnn.predict(inp)
        
        print(f"Predicted {pred}!")
        
        inp = input(">> ")
    
    
        
    
    
    
if __name__ == "__main__":
    main()
    






