import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def get_data_iterator(filepath: str, chunksize: int, cols):
    
    data_batches = tf.data.experimental.make_csv_dataset(
        filepath, batch_size=chunksize,
        label_name=cols[-1],
        select_columns=cols,
        num_epochs=1
    )
    
    return data_batches


def test_train_split(all_dataset, test_data_ratio):
    n = len(all_dataset) + 1
    partition = int(n * test_data_ratio)
    test_dataset = all_dataset.take(partition)
    train_dataset = all_dataset.skip(partition)
    
    return test_dataset, train_dataset




# make_csv_dataset():
# Use like:   
#   for feature_batch, label_batch in data_batches.take(1):
#       print(label_batch)
#       print(feature_batch)
#       bruh = 5





# Doesn't work for large datasets
def get_entire_dataset_as_tfds(file_path: str, use_cols, test_size=0.2): 
    print("Reading CSV file... (1/4)")
    df = pd.read_csv(file_path, usecols=use_cols)
    print("Done.")
    
    print("Converting df to list... (2/4)")
    texts = df[use_cols[0]].tolist()
    labels = df[use_cols[1]].tolist()
    print("Done.")
    
    
    print("Splitting data into training and testing sets... (3/4)")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    print("Done.")
    
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    
    print("Finished loading dataset.")
    
    return train_dataset, test_dataset




