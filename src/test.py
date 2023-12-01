import os


from data import get_data_pipeline

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
CHUNKSIZE = 500


data = get_data_pipeline(STEAM_REVIEWS_SMALL_PATH, 10, USECOLS)



for chunk in data:
    print(chunk)
    
    
print()