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



DATA_FILEPATH = "data/archive/steam_reviews.csv"






# Name of columns to read
usecols = ["review", "recommended"]

# Number of rows to read per chunk
chunksize = 250

# Get iterator
data_gen = get_data_generator(file_path=DATA_FILEPATH, chunk_size=chunksize, use_cols=usecols)


for idx, chunk in enumerate(data_gen):
    if idx == 0:
        print(chunk)
    else:
        break

