import random
import numpy as np
import pandas as pd


def insert_null(df, col, prob):
    """
    insert null value in the dataframe with probability prob

    :param df: pandas dataframe
    :param col: column where put null values
    :param prob: probability to insert null values
    :return: dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doesn't contain a pandas dataframe")
    if col not in df.columns:
        raise ValueError("col doesn't exists in df")
    if prob > 1 or prob < 0:
        raise ValueError("prob must between 0 and 1")

    data = df.copy()
    col = df.columns.get_loc(col)
    ix = [row for row in range(df.shape[0])]
    for row in random.sample(ix, int(round(prob * len(ix)))):
        data.iloc[row, col] = np.NaN
    return data

