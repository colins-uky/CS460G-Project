import pandas as pd


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