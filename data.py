import pandas as pd
import time


def get_data_iterator(filepath):
    start = time.time()

    data = pd.read_csv(
        filepath,
        usecols=[],
        chunksize=500
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