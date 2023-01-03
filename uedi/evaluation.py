import numpy as np
import pandas as pd


def precision_k(ranks, labels, k):
    ids = np.argsort(ranks)[:k]
    np.mean(labels[ids])
    return np.mean(labels[ids])


def random_row(x):
    i = 0 if np.random.randint(2) else -1
    return x.iloc[i]


def prepare_dataset(df: pd.DataFrame, columns: list):
    # Extract left dataframe
    df1 = df[df['source'] == 0]  # keep only one source
    df1 = df1.groupby('entity_id').apply(lambda x: x.iloc[0])  # merge duplicated entities

    # Extract right dataframe
    df2 = df[df['source'] == 1]  # keep only one source
    df2 = df2.groupby('entity_id').apply(lambda x: x.iloc[0])  # merge duplicated entities

    # Extract a possible integrated dataframe
    dfi = df.sort_values('source', ascending=True).groupby('entity_id').apply(lambda x: random_row(x))

    # Filter selected columns
    df1 = df1[columns]  # Left source dataset
    df2 = df2[columns]  # Right source dataset
    dfi = dfi[columns]  # Integrated dataset
    return df1, df2, dfi


def prepare_dataset_match_error(df: pd.DataFrame, columns: list, match_errors: float = 0):
    assert 0 <= match_errors <= 1, 'match errors must be a percentage value'

    # Extract a possible source dataframe
    df1 = df[df['source'] == 0]  # keep only one source
    df1 = df1.groupby('entity_id').apply(lambda x: x.iloc[0])  # merge duplicated entities

    # Extract a possible integrated dataframe
    df2 = df.sort_values('source', ascending=True).groupby('entity_id').apply(lambda x: random_row(x))

    # Create matching errors
    size = int(np.round(match_errors * len(df1)))
    np.random.seed(42)
    order = np.random.permutation(len(df1))
    labels = np.zeros(len(df1), dtype=int)
    labels[order[:size]] = 1

    removed_entities = df1['entity_id'].iloc[order[:size]].values

    cond = df2['entity_id'].isin(removed_entities)
    df2.drop(df2[cond].index, inplace=True)
    df1['label'] = labels.astype(bool)
    df1['isin'] = df1['entity_id'].isin(df2['entity_id'])
    assert np.all(df1['label'] != df1['isin'])

    # Labels represent the record on source dataset that are not in the integrated dataset
    df1 = df1[columns]  # Source dataset with match error
    df2 = df2[columns]  # Integrated dataset
    return df1, df2, labels


def prepare_dataset_concat_error(df: pd.DataFrame, columns: list, concat_error: float = 0):
    assert 0 <= concat_error <= 1, 'concat errors must be a percentage value'

    # Extract source 0
    df1 = df[df['source'] == 0]  # keep only one source
    df1 = df1.groupby('entity_id').apply(lambda x: x.iloc[0])  # merge duplicated entities

    # Extract source 1
    df2 = df[df['source'] == 1]  # keep only one source
    df2 = df2.groupby('entity_id').apply(lambda x: x.iloc[0])  # merge duplicated entities

    # df = df.groupby('entity_id').apply(lambda x: random_row(x))

    # Extract integrated unique entity identifiers
    df_count = df['entity_id'].value_counts()
    entities = df_count.index.values
    size = int(np.round(concat_error * len(entities)))
    order = np.random.permutation(len(entities))
    concat_entities = entities[order[:size]]
    compress_entities = entities[order[size:]]

    expand_entities = [x for x in concat_entities if x in df_count[df_count > 1].index]
    copy_entities = [x for x in concat_entities if x in df_count[df_count == 1].index]

    df_compress = df[df['entity_id'].isin(compress_entities)].groupby('entity_id').apply(lambda x: random_row(x))
    df_expand = df[df['entity_id'].isin(expand_entities)]
    df_copy = df[df['entity_id'].isin(copy_entities)]

    assert len(df_compress) == len(compress_entities)
    assert len(df_compress) == len(df_compress['entity_id'].unique())
    assert df_expand['entity_id'].value_counts().min() > 1
    assert df_copy['entity_id'].value_counts().max() == 1

    labels = np.zeros((len(df_compress) + len(df_expand) + len(df_copy) + len(df_copy)))
    labels[:len(df_compress)] = 0
    labels[len(df_compress):] = 1

    df = pd.concat([df_compress, df_expand, df_copy, df_copy], axis=0, ignore_index=True).reset_index(drop=True)

    for _, row in df_copy.iterrows():
        if row['source'] == 0:
            # df2 = pd.concat([df2, row], axis=0, ignore_index=True)
            df2 = df2.append(row, ignore_index=True)
        elif row['source'] == 1:
            df1 = df1.append(row, ignore_index=True)

    assert df1['entity_id'].value_counts().max() == 1
    assert df2['entity_id'].value_counts().max() == 1

    df1 = df1[columns]
    df2 = df2[columns]
    df = df[columns]
    return df1, df2, df, labels
