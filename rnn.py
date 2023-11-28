# NumPy and Pandas
import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf


# Keras - top level api for tensorflow
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, TextVectorization, LSTM
from keras.optimizers import Adam

from data import get_data_generator


# (7.98 GB FILE WARNING) 
# Use get_data_generator to load data in chunks into memory
DATA_FILEPATH = "data/archive/steam_reviews.csv"



GPT_REVIEWS_PATH = "data/test/gpt_reviews.csv"





# Name of columns to read
#review: text of review, recommended: boolean
USECOLS = ["review", "recommended"]

# Number of rows to read per chunk
CHUNKSIZE = 250

# Get iterator
data_gen = get_data_generator(file_path=GPT_REVIEWS_PATH, chunk_size=10, use_cols=USECOLS)





#





for idx, chunk in enumerate(data_gen):
    if idx == 0:
        print(chunk)
    else:
        break

