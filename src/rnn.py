# Pandas
import pandas as pd

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub


# Keras - top level api for tensorflow
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


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


##### Training Settings #####

# Number of rows to read per chunk
CHUNKSIZE = 100

# 6.4 million rows in dataset / batch size
ESTIMATED_STEPS_PER_DATASET = 6417200 // CHUNKSIZE

STEPS_PER_EPOCH = None
EPOCHS = 10

# Do not ask for prompts 
# (Use at own risk! May cause overwritings!)
FORCE = True

##### Testing Settings #####
TESTING_STEPS = 2500


##### Model Architecture Parameters #####

# Number of distinct words in dataset
VOCAB_SIZE = 1000

EMBEDDING_DIM = 64
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250



####### END GLOBAL VARS #######


####### ESTIMATORS #######

# ~64ms per step if CHUNKSIZE == 100
# ADJUST IF CHANGING CHUNKSIZE
time_per_step = 0.064 #seconds


if STEPS_PER_EPOCH is None:
    time_per_epoch = time_per_step * ESTIMATED_STEPS_PER_DATASET
else:
    time_per_epoch = time_per_step * STEPS_PER_EPOCH


time_per_training = time_per_epoch * EPOCHS

estimated_time_of_completion = time.ctime(time.time() + time_per_training)




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
    text = encoder(text)
    text = tf.expand_dims(text, -1)
    label = tf.expand_dims(label, -1)
    
    return text, label


def filter_input(text, label):
    lowercase = tf.strings.lower(text["review_text"])
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
    # Normalize (-1, 1) labels to (0, 1)
    label = (label + 1) / 2
    
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), ''), label



AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_pipe.map(filter_input)
val_dataset = val_pipe.map(filter_input)
test_dataset = test_pipe.map(filter_input)

train_dataset = train_dataset.map(encode_text)
val_dataset = val_dataset.map(encode_text)
test_dataset = test_dataset.map(encode_text)


train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)




####### END PREPROCESSING #######



####### BUILD MODEL #######

class RNN:
    def __init__(self) -> None:
        
        
     
        self.loss = 0.0
        self.accuracy = 0.0
        self.history = None
        
        self.model = None
        
        self.time_training_finished = None
        
        
    
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
            optimizer=Adam(0.0001),
            metrics=['accuracy']
        )
        
        
        
        
    
        
    def train(self, steps_per_epoch=None, force=False):
        
        print("\n\nStarting training.")
        print(f"Estimated time of completion: {estimated_time_of_completion}.\n\n")
        
        if force:
            answer = 'y'
        else:
            answer = input(f"\nContinue? (y/n) >> ")

        
        if answer.lower() == 'y':
            history = self.model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, validation_steps=30, steps_per_epoch=steps_per_epoch)
            self.time_training_finished = time.localtime(time.time())
        else:
            return -1
        
        
        if force:
            self.force_save()
        
        
        self.history = history
        
        
        
        
    def test(self):
        self.loss, self.accuracy = self.model.evaluate(test_dataset, steps=TESTING_STEPS)
        
        
    def print_score(self):
        print()
        print(f"Loss:       {self.loss:.5f}")
        print()
        print(f"Accuracy:   {self.accuracy:.5f}")
        
        
        
    def predict(self, inp: str):
        with tf.device("/CPU:0"):
            text, _ = filter_input({"review_text": inp}, 0)
            text = encoder([inp])
            text = tf.expand_dims(text, -1)
        
            raw_pred = self.model.predict(text)
        
        pred = raw_pred.tolist()[0][0]
        
        if pred >= 0.5: # positive prediction
            label = "positive"
            probability = pred * 100
        else:
            label = "negative"
            probability = (1 - pred) * 100
        
        return label, probability
        
  
        
    def prompt_save(self):
        cwd = os.path.join(CWD)
        filepath = os.path.join(cwd, "bin", "rnn.keras")

        answer = input(f"\nSave model to {filepath} ? \n(y/n) >> ")
        
        if answer.lower() == 'y':
            self.model.save(filepath)
        else:
            print("Canceled save.")
            
    def force_save(self):
        cwd = os.path.join(CWD)
        if self.time_training_finished is None:
            time_struct = time.localtime(time.time())
        else:
            time_struct = self.time_training_finished
        
        filename = f"model_{time_struct.tm_mon}_{time_struct.tm_mday}_{time_struct.tm_year}-{time_struct.tm_hour}_{time_struct.tm_min}_{time_struct.tm_sec}.keras"
        filepath = os.path.join(cwd, "bin", "autosaved", filename)

        print(f"Force saving model to {filepath}")
        
        self.model.save(filepath)
        
            
        
    def load_from_disk(self, filepath=None, force=False):
        cwd = os.path.join(CWD)
        
        if filepath is None:
            filepath = os.path.join(cwd, "bin", "rnn.keras")

        if force:
            answer = 'y'
        else:
            answer = input(f"\nLoad model from {filepath} ? \n(y/n) >> ")
        
        if answer.lower() == 'y':
            print(f"Loading model from {filepath}")
            self.model = load_model(filepath)
        else:
            print("Canceled load.")
            
            
    def question_loop(self):
        print("RNN Input Loop:\nType reviews or 'q' to quit.\n")
        inp = input("\n>> ")
        while inp != 'q':
            label, prob = self.predict(inp)
            
            print(f"Predicted {label} sentiment with {int(prob)}% certainty.")
            
            inp = input("\n>> ")
            
            
    def save_history_to_file(self):
        history_df = pd.DataFrame(self.history.history)
        
        
        filename = f"model_training_history{self.time_training_finished.tm_mon}_{self.time_training_finished.tm_mday}_{self.time_training_finished.tm_year}-{self.time_training_finished.tm_hour}_{self.time_training_finished.tm_min}_{self.time_training_finished.tm_sec}.json"
        filepath = os.path.join(CWD, "bin", "history", filename)
        
        # save to json:  
        
        with open(filepath, mode='w') as file:
            history_df.to_json(file)

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
    
    #rnn.build()
    
    rnn.load_from_disk(filepath=r"C:\Users\colin\OneDrive\Desktop\Repositories\CS460G-Project\bin\rnn.keras", force=FORCE)
    
    #rnn.train(steps_per_epoch=STEPS_PER_EPOCH, force=FORCE)
    #rnn.save_history_to_file()
    
    #rnn.test()
    
    #rnn.print_score()
    
    
    
    rnn.question_loop()
    
    
        
    
    
    
if __name__ == "__main__":
    main()
    






