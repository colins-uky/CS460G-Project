import numpy as np
from numpy.random import seed
seed(1)

import matplotlib.pyplot as plt
import pandas as pd
import string
import os
import shutil
import re

# Tensorflow
import tensorflow as tf


# Keras - top level api for tensorflow
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, TextVectorization
from keras.optimizers import Adam


#generate sequence
sequence = np.array(list(range(10))).astype(float)

#number of training words
window_size = 3


#generate training data
train_data = [np.array(sequence[i:i+window_size]).reshape(1,-1) for i in range(sequence.shape[0]-window_size)]
train_data = np.concatenate(train_data*10, axis=0)
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)


#get training labels
train_label = [sequence[i+window_size] for i in range(sequence.shape[0]-window_size)]
train_label = np.concatenate([train_label]*10).reshape(-1,1)



# Initialize Model
model = Sequential()

#RNN
model.add(SimpleRNN(1, activation="linear"))


# Compile 
opt = Adam(learning_rate=1)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])


# Train
model.fit(train_data, train_label, epochs=100, verbose=1)


model.summary()