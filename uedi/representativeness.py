import numpy as np
import pandas as pd
from .tokenizer import tokenizer
from .model import get_distribution


def representativness_difference(dist_s: pd.Series, dist_i: pd.Series) -> [pd.Series, pd.Series]:
    """ Given the distributions of the source and the integrated dataset
    it will compute source and integrated representativness difference for source vocabs """
    # Remove zero score from source distribution
    dist_s = dist_s[dist_s != 0]

    # Extract vocabs and scores from source dataset distribution
    vocabs, scores = dist_s.index.values, dist_s.values
    s1 = np.asarray(scores, dtype=float)  # source dataset scores

    # Get scores from integrated dataset distribution for vocabs inside the source dataset
    s2 = [dist_i[w] if w in dist_i else 0 for w in dist_s.index]
    s2 = np.asarray(s2, dtype=float)  # integrated dataset scores

    # Compute minimum and maxima intersection
    minima = np.minimum(s1, s2)
    maxima = np.maximum(s1, s2)

    # Compute difference
    s_diff = s1 - minima
    i_diff = s2 - minima

    # Normalize
    s_diff = 1 - s_diff / maxima
    i_diff = 1 - i_diff / maxima

    s_diff = pd.Series(s_diff, index=vocabs)
    i_diff = pd.Series(i_diff, index=vocabs)

    return s_diff, i_diff


def representativness_score(dist_s: pd.Series, dist_i: pd.Series) -> [float, float]:
    """ Given the distributions of the source and the integrated dataset
    it will compute input and output representativness """
    s_diff, i_diff = representativness_difference(dist_s, dist_i)

    # Compute average
    input_repr = np.mean(s_diff)
    output_repr = np.mean(i_diff)
    return input_repr, output_repr


def representativness(df_s: pd.DataFrame, df_i: pd.DataFrame):
    """ Given the source and the integrated dataset it will compute input and output representativness """
    # Tokenize each row in the dataframes
    strategy = 'stemming'  # stemming, lemmatization, None
    record_s = [tokenizer(row, strategy=strategy) for _, row in df_s.iterrows()]
    record_i = [tokenizer(row, strategy=strategy) for _, row in df_i.iterrows()]

    # Compute distribution of the tokenized dataframes
    dist_s = get_distribution(record_s)
    dist_i = get_distribution(record_i)

    # Compute input and output representativness
    input_repr, output_repr = representativness_score(dist_s, dist_i)
    return input_repr, output_repr


