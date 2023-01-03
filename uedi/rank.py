from typing import List
import numpy as np
import pandas as pd
from .tokenizer import tokenizer
from .model import get_distribution
from .representativeness import representativness_difference


def ranking_score(record_dist: pd.Series, repr_dist: pd.Series):
    vocabs = record_dist.index
    r = None
    if repr_dist.index.isin(vocabs).any():
        r = np.mean(repr_dist[repr_dist.index.isin(vocabs)])
    return r


def coverage_score(record_dist: pd.Series, repr_dist: pd.Series):
    vocabs = record_dist.index
    c = repr_dist.index.isin(vocabs).sum()
    return c


def input_ranking(df_s: pd.DataFrame, df_i: pd.DataFrame):
    # Tokenize each row in the dataframes
    strategy = 'stemming'
    record_s = [tokenizer(row, strategy=strategy) for _, row in df_s.iterrows()]
    record_i = [tokenizer(row, strategy=strategy) for _, row in df_i.iterrows()]

    # Compute distribution of the tokenized dataframes
    dist_s = get_distribution(record_s)
    dist_i = get_distribution(record_i)

    # Compute input and output representativness
    s_diff, i_diff = representativness_difference(dist_s, dist_i)

    # Ranking source record
    dist_record_s = [get_distribution([row]) for row in record_s]
    input_ranks = [ranking_score(row, s_diff) for row in dist_record_s]
    input_ranks = np.array(input_ranks)

    return input_ranks


def output_ranking(df_list: List[pd.DataFrame], df_i: pd.DataFrame):
    # Tokenize each row in the dataframes
    strategy = 'stemming'

    # Compute integrated distribution
    record_i = [tokenizer(row, strategy=strategy) for _, row in df_i.iterrows()]
    dist_i = get_distribution(record_i)

    ranking_list = []
    coverage_list = []

    for df_s in df_list:
        record_s = [tokenizer(row, strategy=strategy) for _, row in df_s.iterrows()]
        # Compute distribution of the tokenized dataframes
        dist_s = get_distribution(record_s)

        # Compute input and output representativness
        s_diff, i_diff = representativness_difference(dist_s, dist_i)

        # Ranking integration record
        dist_record_i = [get_distribution([row]) for row in record_i]
        output_ranks = [ranking_score(row, i_diff) for row in dist_record_i]
        output_coverage = [coverage_score(row, i_diff) for row in dist_record_i]

        ranking_list.append(output_ranks)
        coverage_list.append(output_coverage)

    ranking_list = np.array(ranking_list).T
    coverage_list = np.array(coverage_list).T

    # pos = np.argmax(coverage_list, axis=1)
    # output_ranks = []
    # for i, idx in enumerate(pos):
    #     output_ranks.append(ranking_list[i, idx])

    output_ranks = []
    for i in range(len(coverage_list)):
        df_rank = pd.DataFrame({'Coverage': coverage_list[i, :], 'Rank': ranking_list[i, :]})
        df_rank = df_rank.sort_values(['Rank'], ascending=False)
        output_ranks.append(df_rank['Rank'].iloc[0])

    return output_ranks
