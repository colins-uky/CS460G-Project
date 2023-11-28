import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split



def get_data_iterator(filepath: str, chunksize: int, cols):
    
    data_batches = tf.data.experimental.make_csv_dataset(
        filepath, batch_size=chunksize,
        label_name=cols[-1],
        select_columns=cols
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



get_data_iterator("data/test/gpt_reviews.csv", 16, ['review', 'recommended'])






def get_data_generator(file_path: str, chunk_size: int, use_cols):

    data = pd.read_csv(
        file_path,
        usecols=use_cols,
        chunksize=chunk_size
    )

    return data


# Use data like...
#
#   for idx, chunk in enumerate(data):
#       if idx == 0:
#           print(chunk)
#       else:
#           break
# 



# Doesn't work for large datasets
def get_entire_dataset_as_tfds(file_path: str, use_cols): 
    print("Reading CSV file... (1/4)")
    df = pd.read_csv(file_path, usecols=use_cols)
    print("Done.")
    
    print("Converting df to list... (2/4)")
    texts = df['review'].tolist()
    labels = df['recommended'].tolist()
    print("Done.")
    
    
    print("Splitting data into training and testing sets... (3/4)")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print("Done.")

    print("Converting testing and training sets to tf Datasets... (4/4)")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
    print("Done.")
    
    print("Finished loading dataset.")
    
    return train_dataset, test_dataset
