import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
from gensim.parsing.preprocessing import preprocess_string
from sklearn.metrics import confusion_matrix
from uedi.utils.general_utilities import check_parameter_type, check_cols_in_dataframe
from uedi.utils.file_utilities import check_file_existence, check_multiple_file_existence
from uedi.data_integration.data_fusion import DataFusionComponent
from uedi.data_integration.data_preparation import convert_matching_pairs_to_integrated_dataset


def check_document_format(doc: list, param_name: str):
    """
    This function checks the format of a document.

    :param doc: document whose format has to be checked
    :param param_name: name of the document parameter
    :return: None
    """

    check_parameter_type(doc, param_name, list, 'list')

    for word in doc:
        check_parameter_type(word, "{} word".format(param_name), str, 'string')


def check_tokenized_dataset_format(dataset: np.ndarray, param_name: str):
    """
    This function checks the format of a tokenized dataset.

    :param dataset: tokenized dataset whose format has to be checked
    :param param_name: name of the dataset parameter
    :return: None
    """
    check_parameter_type(dataset, param_name, np.ndarray, 'numpy array')

    for doc in dataset:
        check_document_format(doc, "{} document".format(param_name))


def get_source(df: pd.DataFrame, sid: int):
    """
    This function extracts from the provided integrated dataset the information belonging to the specified data source.

    :param df: Pandas DataFrame containing an integrated dataset
    :param sid: data source identifier
    :return: Pandas DataFrame containing only the information belonging to the specified data source
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(sid, 'sid', int, 'integer')

    cond = df['source'] == sid
    if cond.any():
        return df.loc[cond, :].iloc[0, :]
    else:
        return df.iloc[0, :]


def get_random(df: pd.DataFrame):
    """
    This function extracts a random row from the input Pandas DataFrame.

    :param df: Pandas DataFrame object
    :return: Pandas DataFrame containing only the selected row
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    np.random.seed(24)
    idx = np.random.randint(0, len(df))
    return df.iloc[idx, :]


def get_integrated_s1_s2(integrated_df: pd.DataFrame, option: int = 1):
    """
    This function applies some simple data fusion techniques on the input integrated dataset.

    :param integrated_df: Pandas DataFrame containing an integrated dataset
    :param option: integer flag to select the data fusion technique (1 = data source 0, 2 = data source 1, 3 = mix of
                   records from data source 0 and 1).
    :return: Pandas DataFrame containing a compression version of the input integrated dataset
    """

    check_parameter_type(integrated_df, 'integrated_df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(option, 'option', int, 'integer')

    if option == 1:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x : get_source(x, sid=0))
    elif option == 2:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x : get_source(x, sid=1))
    else:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x : get_random(x))

    return integrated_s1_s2


def get_sources(integrated_ds: pd.DataFrame):
    """
    This function extracts single data sources information from the provided integrated dataset.

    :param integrated_ds: Pandas DataFrame containing an integrated dataset
    :return: (Pandas DataFrame for the source A, Pandas DataFrame for the source B)
    """

    check_parameter_type(integrated_ds, 'integrated_ds', pd.DataFrame, 'Pandas DataFrame')

    left_columns = [col for col in integrated_ds.columns if 'left_' in col]
    right_columns = [col for col in integrated_ds.columns if 'right_' in col]

    ds_a = integrated_ds[left_columns].copy()
    ds_a.columns = [col.replace('left_', '') for col in ds_a.columns]
    ds_a.drop_duplicates(inplace=True)
    ds_a.reset_index(inplace=True, drop=True)

    ds_b = integrated_ds[right_columns].copy()
    ds_b.columns = [col.replace('right_', '') for col in ds_b.columns]
    ds_b.drop_duplicates(inplace=True)
    ds_b.reset_index(inplace=True, drop=True)

    return ds_a, ds_b


def dataset_attribute_tokenizer(df: pd.DataFrame, columns: list = None):
    """
    This function applies an attribute-tokenization to the input Pandas DataFrame.

    :param df: Pandas DataFrame object to be tokenized
    :param columns: optional column names where to limit the tokenization
    :return: tokenized version of the input Pandas DataFrame: list of lists of the DataFrame rows
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

    if columns is not None:
        check_cols_in_dataframe(df, columns)

    if columns is not None:
        res = df[columns]
    else:
        res = df

    res = np.array(res.astype(str).apply(lambda x: [' '.join(x)], axis=1).to_list())

    return res


def dataset_tokenizer(df: pd.DataFrame, columns: list = None):
    """
    This function applies a tokenization to the input Pandas DataFrame.

    :param df: Pandas DataFrame object to be tokenized
    :param columns: optional column names where to limit the tokenization
    :return: tokenized version of the input Pandas DataFrame: list of lists of the DataFrame rows
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

    if columns is not None:
        check_cols_in_dataframe(df, columns)

    if columns is not None:
        res = df[columns]
    else:
        res = df

    res = np.array(res.astype(str).apply(lambda x: preprocess_string(' '.join(x)), axis=1).to_list())

    return res


# FIXME: is it possible to use a public implementation or do we have data format issues?
def jaccard_distance(r1, r2):
    r1 = r1[0].split()
    r2 = r2[0].split()
    s1 = set(r1)
    s2 = set(r2)
    return 1 - len(s1.intersection(s2)) / len(s1.union(s2))


def get_duplicate(df: pd.DataFrame, th: float = 0.4, columns: list = None):
    """
    This function retrieves the duplicated records from the input DataFrame.
    Records with a Jaccard distance smaller the the input threshold are considered as duplicates.

    :param df: Pandas DataFrame object where to extract duplicated records
    :param th: threshold value for the Jaccard distance
    :param columns: optional column names where to focus in the process
    :return: (set of duplicated records)
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
    check_parameter_type(th, 'th', float, 'float')

    if th < 0 or th > 1:
        raise ValueError("Wrong threshold value.")

    if columns is not None:
        check_cols_in_dataframe(df, columns)

    # tokenize the dataset
    records = dataset_attribute_tokenizer(df, columns=columns)

    # compute the pair-wise Jaccard distances
    compact_dm = pdist(records, metric=jaccard_distance)
    compact_dm = squareform(compact_dm)

    # get duplicated records
    pos = np.arange(len(records))
    delete_set = set()
    for i in range(len(records)):
        dvec = compact_dm[i]

        indexes = pos[dvec < th]
        if len(indexes) > 1:
            for j in indexes:
                if j != i and j > i:
                    delete_set.add(j)

    return delete_set


def keep_unique_records_from_data_source(integrated_ds: pd.DataFrame, source_ds: pd.DataFrame, left_match: str,
                                         right_match: str, suffix: str, th: float, columns: list = None):
    """
    This function removes from the input integrated dataset the duplicated records coming from the input data source
    dataset.

    :param integrated_ds: Pandas DataFrame object containing the integrated dataset
    :param source_ds: Pandas DataFrame object containing the data source dataset
    :param left_match: column of the integrated dataset that points to the primary key of the data source dataset
    :param right_match: primary key column of the data source dataset
    :param suffix: attribute suffix used in the integrated dataset to refer to data source attributes
    :param th: threshold value for finding duplicates
    :param columns: optional data source column names where to focus the process
    :return: cleaned version of the integrated dataset where duplicated records coming from the data source have been
             removed
    """

    # check parameter data types
    check_parameter_type(integrated_ds, 'integrated_ds', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(source_ds, 'source_ds', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(left_match, 'left_match', str, 'string')
    check_parameter_type(right_match, 'right_match', str, 'string')
    check_parameter_type(suffix, 'suffix', str, 'string')
    check_parameter_type(th, 'th', float, 'float')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

    # check parameter data values
    check_cols_in_dataframe(integrated_ds, [left_match])
    check_cols_in_dataframe(source_ds, [right_match])

    if th < 0 or th > 1:
        raise ValueError("Wrong threshold value.")

    if columns is not None:
        check_cols_in_dataframe(source_ds, columns)

    pos = np.arange(len(source_ds))

    # get duplicated records from the data sources
    # FIXME: threshold
    duplicate_set = get_duplicate(source_ds, th=0.4, columns=columns)
    print('Found {} duplicated records from a data source.'.format(len(duplicate_set)))

    # get unique records
    list_keep = set(pos).difference(duplicate_set)
    list_keep = list(list_keep)
    cleaned_ds = integrated_ds.merge(source_ds[[right_match]].iloc[list_keep],
                                     left_on=left_match, right_on=right_match,
                                     suffixes=(False, suffix))

    return cleaned_ds


# FIXME: is this function duplicated?
def clean_single_data(source: pd.DataFrame, th: float, columns: list = None):
    """
    This function cleans a single data source.

    :param source: Pandas DataFrame containing the information of the data source
    :param th: threshold for cleaning process
    :param columns: optional list of columns where to limit the cleaning process
    :return: Pandas DataFrame containing a cleaned version of the input data source
    """

    check_parameter_type(source, 'source', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(th, 'th', float, 'float')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

    if th < 0 or th > 1:
        raise ValueError("Wrong threshold value.")

    if columns is not None:
        check_cols_in_dataframe(source, columns)

    pos = np.arange(len(source))
    duplicate_set = get_duplicate(source, th, columns=columns)
    list_keep = set(pos).difference(duplicate_set)
    list_keep = list(list_keep)
    source = source.iloc[list_keep]

    return source


def clean_integrated_data(filename: str, th: float, columns: list = None):
    """
    This function removes duplicated entities from the provided integrated dataset.

    :param filename: name of the integrated dataset to be deduplicated
    :param th: the threshold value for the deduplication process
    :param columns: optional column names to be considered in the deduplication process
    :return: Pandas DataFrame object containing the cleaned integrated dataset
    """

    # check parameter data types
    check_parameter_type(filename, 'filename', str, 'string')
    check_parameter_type(th, 'th', float, 'float')
    check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

    # check parameter data values
    check_file_existence(filename)

    integrated_ds = pd.read_csv(filename)

    if th < 0 or th > 1:
        raise ValueError("Wrong threshold value.")

    print('Removing duplicates from {} (th={})'.format(filename.split(os.sep)[-1], th))

    # extract data source information
    ds_a, ds_b = get_sources(integrated_ds)

    # clean data source b
    cleaned_ds = keep_unique_records_from_data_source(integrated_ds, ds_b, left_match='right_id', right_match='id',
                                                      suffix='_right', th=th, columns=columns)

    # clean data source a
    cleaned_ds = keep_unique_records_from_data_source(cleaned_ds, ds_a, left_match='left_id', right_match='id',
                                                      suffix='_left', th=th, columns=columns)

    print("Duplicates removal completed successfully.")

    return cleaned_ds[['ltable_id', 'rtable_id', 'label']]


def expand_integrated_dataset(integrated_path: str, source_a_path: str, source_b_path: str):
    """
    This function expands the integrated dataset with the records from source_a and source_b.

    :param integrated_path: integrated dataset file path
    :param source_a_path: source a file path
    :param source_b_path: source b file path
    :return: Pandas DataFrame containing an expanded version of the integrated dataset
    """

    check_parameter_type(integrated_path, 'integrated_path', str, 'string')
    check_parameter_type(source_a_path, 'source_a_path', str, 'string')
    check_parameter_type(source_b_path, 'source_b_path', str, 'string')
    check_multiple_file_existence([integrated_path, source_a_path, source_b_path])

    ds = pd.read_csv(integrated_path)
    ds_a = pd.read_csv(source_a_path)
    ds_b = pd.read_csv(source_b_path)

    assert 'ltable_id' in ds
    assert 'rtable_id' in ds
    assert 'id' in ds_b
    assert 'id' in ds_a

    ds_a = ds_a.add_prefix('left_')
    ds_b = ds_b.add_prefix('right_')

    ds = pd.merge(ds, ds_a, how='inner', left_on='ltable_id', right_on='left_id', suffixes=(False, False))
    ds = pd.merge(ds, ds_b, how='inner', left_on='rtable_id', right_on='right_id', suffixes=(False, False))

    ds['id'] = np.arange(1, len(ds) + 1)

    return ds


def get_three_sources(df: pd.DataFrame):
    """
    This function generates 3 datasets starting from the input integrated dataset.

    :param df: Pandas DataFrame containing the integrated dataset to be split
    :return: three Pandas DataFrame objects
    """

    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')

    # print('{} entities in integrated dataframe with {} records'.format(len(df['entity_id']. unique()), len(df)))

    match_data = df.groupby('entity_id').filter(lambda x: len(x) > 1)
    non_match_data = df.groupby('entity_id').filter(lambda x: len(x) == 1)
    # print("Size match data: {}".format(len(match_data)))
    # print("Size non-match data: {}".format(len(non_match_data)))

    s1 = match_data[match_data['source'] == 0].reset_index(drop=True)
    s2 = match_data[match_data['source'] == 1].reset_index(drop=True)
    s1 = s1.loc[~s1['entity_id'].duplicated(keep='last')]
    s2 = s2.loc[~s2['entity_id'].duplicated(keep='last')]
    # print("Max s1 size: {}".format(len(s1)))
    # print("Max s2 size: {}".format(len(s2)))

    size_multiple_of = 100
    if len(s1) < 100:
        size_multiple_of = 10

    s1_size = int(len(s1) / size_multiple_of) * size_multiple_of
    s2_size = int(s1_size / 2)
    # print("Selected s1 size: {}".format(s1_size))
    # print("Selected s2/s3 size: {}".format(s2_size))

    s1 = s1.iloc[:s1_size]
    s2 = s2.iloc[:s2_size]
    s3 = non_match_data.iloc[:s2_size].reset_index(drop=True)
    # print('{} entities in s1 with {} records'.format(len(s1['entity_id'].unique()), len(s1)))
    # print('{} entities in s2 with {} records'.format(len(s2['entity_id'].unique()), len(s2)))
    # print('{} entities in s3 with {} records'.format(len(s3['entity_id'].unique()), len(s3)))
    print("|S1| = {}".format(len(s1)))
    print("|S2| = {}".format(len(s2)))
    print("|S3| = {}".format(len(s3)))

    return s1, s2, s3


def get_integration_sample(df: pd.DataFrame, label_col: str, num_matches: int, num_non_matches: int,
                           random_state: int = 24):
    """
    This function creates a sample of the input integrated dataset by extracting an user-provided number of matching and
    non-matching pairs.

    :param df: Pandas DataFrame object containing the integrated dataset
    :param label_col: integrated dataset column containing the label of the pair of records (i.e., match or non-match)
    :param num_matches: number of matching pairs to be sampled
    :param num_non_matches: number of non-matching pairs to be sampled
    :param random_state: seed for random choices
    :return: Pandas DataFrame object containing a sample of the input integrated dataset
    """

    # check input parameter data types
    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(label_col, 'label_col', str, 'string')
    check_parameter_type(num_matches, 'num_matches', int, 'integer')
    check_parameter_type(num_non_matches, 'num_non_matches', int, 'integer')
    check_parameter_type(random_state, 'random_state', int, 'integer')

    # check input parameter data values
    check_cols_in_dataframe(df, [label_col])

    match_records = df[df[label_col] == 1]
    if num_matches <= 0 or num_matches > len(match_records):
        raise ValueError("Wrong value for parameter num_matches. Only values in the range (1, {}) are allowed.".format(
            len(match_records)))

    non_match_records = df[df[label_col] == 0]
    if num_non_matches <= 0 or num_non_matches > len(non_match_records):
        raise ValueError(
            "Wrong value for parameter num_non_matches. Only values in the range (1, {}) are allowed.".format(
                len(non_match_records)))

    # FIXME: matching records that refer to the same entity can be extracted. is it a problem?
    sample_matches = match_records.sample(num_matches, random_state = random_state)

    # FIXME: there can be an overlap between non_matches and matches
    sample_non_matches = non_match_records.sample(num_non_matches, random_state = random_state)

    sample = pd.concat([sample_matches, sample_non_matches])

    return sample


def generate_matching_predictions(dataset: pd.DataFrame, match_col: str, random_state: int = 24,
                                  match_ratio: float = None, non_match_ratio: float = None):
    """
    This function generates synthetic matching predictions by:
    1) randomly selecting a certain number of matches (i.e. the match_ratio parameter) and/or non_matches
       (i.e. the non_match_ratio parameter) from the input dataset
    2) flipping the not selected matches and/or non_matches

    :param dataset: Pandas DataFrame object containing the integrated dataset
    :param match_col: the dataset column which contains the true match/non-match labels
    :param random_state: the seed for random choices
    :param match_ratio: the percent of exact matches to extract from the dataset
    :param non_match_ratio: the percent of exact non-matches to extract from the dataset
    :return: Pandas DataFrame containing the original dataset with a new column 'pred' with the synthetic predictions
    """

    # check input parameters data types
    check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(match_col, 'match_col', str, 'string')
    check_parameter_type(random_state, 'random_state', int, 'integer')
    check_parameter_type(match_ratio, 'match_ratio', float, 'float', optional_param=True)
    check_parameter_type(non_match_ratio, 'non_match_ratio', float, 'float', optional_param=True)

    # check input parameters data values
    check_cols_in_dataframe(dataset, [match_col])

    if match_ratio is not None:
        if match_ratio < 0 or match_ratio > 1:
            raise ValueError("Wrong match ratio value. Only values in the range [0, 1] are allowed.")

    if non_match_ratio is not None:
        if non_match_ratio < 0 or non_match_ratio > 1:
            raise ValueError("Wrong non match ratio value. Only values in the range [0, 1] are allowed.")

    # select all the matching and non-matching data from the input dataset
    match_col_values = dataset[match_col]
    match_data = match_col_values[match_col_values == 1]
    # FIXME: the following instruction could select also matching entities
    non_match_data = match_col_values[match_col_values == 0]
    # print("\tNum. match data: {}".format(len(match_data)))
    # print("\tNum. non-match data: {}".format(len(non_match_data)))

    # select the user-provided ratio of matching and/or non-matching data
    if match_ratio is not None:
        match = match_data.copy()
        match_ids = match.sample(frac=match_ratio, random_state=random_state).index.values
        # mark the not selected matches as non match
        match.loc[~match.index.isin(match_ids)] = 0

    else:
        match = match_data.copy()

    if non_match_ratio is not None:
        non_match = non_match_data.copy()
        non_match_ids = non_match.sample(frac=non_match_ratio, random_state=random_state).index.values
        # mark the not selected non-matches as match
        non_match.loc[~non_match.index.isin(non_match_ids)] = 1
    else:
        non_match = non_match_data.copy()

    preds = pd.concat([match, non_match])
    dataset["pred"] = preds

    return dataset


def create_integration_dataset_by_data_type(df: pd.DataFrame, target_col: str, data_type: str = 'all',
                                            random_seed: int = 24):
    """
    This function takes in input a dataset containing some matching pairs records, optionally selects only the
    matching/non-matching data and convert it in an integrated dataset.

    :param df: Pandas DataFrame containing the matching pairs
    :param target_col: the column of the input dataset that contains the label to detect match/non_match data
    :param data_type: the type of data to consider (i.e., match, non_match or all)
    :param random_seed: the seed for random choices
    return: (data from source 1, data from source 2, integrated dataset)
    """

    # check input parameter data types
    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(target_col, 'target_col', str, 'string')
    check_parameter_type(data_type, 'data_type', str, 'string')
    check_parameter_type(random_seed, 'random_seed', int, 'integer')

    check_cols_in_dataframe(df, [target_col])

    # check input parameter data values
    data_types = ["match", "non_match", "all"]
    if data_type not in data_types:
        raise ValueError("Wrong data values for parameter data_type. Only values {} are allowed.".format(data_types))

    y_pred = df[target_col]

    if data_type == "match":
        data = df.loc[y_pred == 1].copy()

    elif data_type == "non_match":
        data = df.loc[y_pred == 0].copy()

    elif data_type == "all":
        data = df.copy()

    print('\nDataset {}: {}'.format(data_type, len(data)))

    ds_a, ds_b = get_sources(data)
    print('LEN {}(A): {}'.format(data_type, len(ds_a)))
    print('LEN {}(B): {}'.format(data_type, len(ds_b)))

    # params for the first data source
    ds_a_id = 0
    ds_a_index_col = "id"

    # params for the second data source
    ds_b_id = 1
    ds_b_index_col = "id"

    # params for the matching pairs dataset
    match_id = "id"
    left_id = "left_id"
    right_id = "right_id"
    match_label = target_col

    # convert the matching pairs dataset in an integrated dataset
    data1, data2, integration_data = convert_matching_pairs_to_integrated_dataset(ds_a, ds_a_id, ds_a_index_col, ds_b,
                                                                                  ds_b_id, ds_b_index_col,
                                                                                  data, match_id, left_id, right_id,
                                                                                  match_label)

    return data1, data2, integration_data


def create_compact_integration_dataset_by_data_type(df: pd.DataFrame, data_type: str, target_col: str,
                                                    random_seed: int = 24):
    """
    This function takes in input a dataset containing some matching pairs records, optionally selects only the
    matching/non-matching data and convert it in a compact integrated dataset (i.e., it applies data fusion to solve
    matched records).

    :param df: Pandas DataFrame containing the matching pairs
    :param data_type: the type of data to consider (i.e., match, non_match or all)
    :param target_col: the column of the input dataset that contains the label to detect match/non_match data
    :param random_seed: the seed for random choices
    return: (data from source 1, data from source 2, integrated dataset)
    """

    # check input parameter data types
    check_parameter_type(df, 'df', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(data_type, 'data_type', str, 'string')
    check_parameter_type(target_col, 'target_col', str, 'string')
    check_parameter_type(random_seed, 'random_seed', int, 'integer')

    # check input parameter data values
    data_types = ["match", "non_match", "all"]
    if data_type not in data_types:
        raise ValueError("Wrong data values for parameter data_type. Only values {} are allowed.".format(data_types))

    check_cols_in_dataframe(df, [target_col])

    data1, data2, integration_data = create_integration_dataset_by_data_type(df, target_col, data_type=data_type,
                                                                             random_seed=random_seed)

    # apply a data fusion process to the integration dataset
    data_fusion_comp = DataFusionComponent(integration_data)
    integrated_dataset = data_fusion_comp.select_random_records(random_seed)

    return data1, data2, integrated_dataset


def get_wrong_categorized_entities(s: pd.DataFrame, merge: pd.DataFrame, s_pk: str):
    """
    This function identifies the new entities included in the source s (i.e., the entities that don't match in the ground
    truth (column label) with any other entity) and among them discovers the wrong-categorized entities (i.e., the new
    entities recognized as old entities). These errors are inserted in a confusion matrix.

    :param s: data source
    :param merge: integrated dataset containing true labels and predicted labels
    :param s_pk: the primary key column of the data source
    return: Numpy and Pandas versions of the confusion matrix
    """

    check_parameter_type(s, 's', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(merge, 'merge', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(s_pk, 's_pk', str, 'string')

    check_cols_in_dataframe(merge, [s_pk])

    s['true_new'] = s.apply(lambda x: merge[(merge[s_pk] == x['id']) & (merge['label'] == 1)].empty, axis=1).astype(int)
    s['pred_new'] = s.apply(lambda x: merge[(merge[s_pk] == x['id']) & (merge['pred'] == 1)].empty, axis=1).astype(int)
    cm = confusion_matrix(s['true_new'].values, s['pred_new'].values)
    df_cm = pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1'])

    return cm, df_cm


def get_concat_and_compression_scores(s1, s2, merge):
    """
    This function discovers the entities of the two input data sources (i.e., s1 and s2) that are wrong-categorized
    and measures for each pair (source, integrated dataset) the extent of these errors via a confusion matrix.

    :param s1: first data source
    :param s2: second data source
    :param merge: integrated dataset containing true labels and predicted labels
    """

    check_parameter_type(s1, 's1', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(s2, 's2', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(merge, 'merge', pd.DataFrame, 'Pandas DataFrame')

    cm_s1, df_cm_s1 = get_wrong_categorized_entities(s1, merge, 'ltable_id')
    # true_match_s1, false_concat_s1, false_match_s1, true_concat_s1 = cm_s1.ravel()
    # print("\nConfusion matrix respect S1:\n{}".format(df_cm_s1))

    cm_s2, df_cm_s2 = get_wrong_categorized_entities(s2, merge, 'rtable_id')
    # true_match_s2, false_concat_s2, false_match_s2, true_concat_s2 = cm_s2.ravel()
    # print("\nConfusion matrix respect S2:\n{}".format(df_cm_s2))

    # concat_increment = (false_concat_s1 + false_concat_s2) / (df_cm_s1["pred_1"].sum() + df_cm_s2["pred_1"].sum())
    # print("\nConcat increment: {}".format(concat_increment))

    # compression_increment = (false_match_s1 + false_match_s2) / (df_cm_s1["pred_0"].sum() + df_cm_s2["pred_0"].sum())
    # print("\nCompression increment: {}\n".format(compression_increment))

    return cm_s1, cm_s2


def get_effectiveness_metrics_from_confusion_matrix(cm: np.ndarray, class_labels: dict = None):
    """
    This function computes some simple effectiveness metrics over the provided confusion matrix.

    :param cm: the confusion matrix
    :param: optional dictionary containing the names of the classes
    :return: dictionary containing the computed effectiveness metrics
    """

    check_parameter_type(cm, 'cm', np.ndarray, 'Numpy array')
    check_parameter_type(class_labels, 'class_labels', dict, 'Dictionary', optional_param=True)

    if cm.shape != (2, 2):
        raise ValueError("Wrong value for parameter cm. Only binary confusion matrices are allowed.")

    if class_labels is not None:
        if len(class_labels) != 2:
            raise ValueError("Wrong value for parameter class_labels.")

        if '0' not in class_labels and '1' not in class_labels:
            raise ValueError("Wrong value for parameter class_labels.")

    true_positive, false_negative, false_positive, true_negative = cm.ravel()
    positive_prec = true_positive / (true_positive + false_positive)
    positive_recall = true_positive / (true_positive + false_negative)
    negative_prec = true_negative / (true_negative + false_negative)
    negative_recall = true_negative / (true_negative + false_positive)

    positive_class_label = 'positive'
    negative_class_label = 'negative'
    if class_labels is not None:
        positive_class_label = class_labels['1']
        negative_class_label = class_labels['0']

    metrics_map = {'true_{}'.format(positive_class_label): true_positive,
                   'false_{}'.format(negative_class_label): false_negative,
                   'false_{}'.format(positive_class_label): false_positive,
                   'true_{}'.format(negative_class_label): true_negative,
                   '{}_prec'.format(positive_class_label): positive_prec,
                   '{}_rec'.format(positive_class_label): positive_recall,
                   '{}_prec'.format(negative_class_label): negative_prec,
                   '{}_rec'.format(negative_class_label): negative_recall}

    return metrics_map


def get_match_non_match_sizes(dataset: pd.DataFrame, label_col: str):
    """
    This function counts the number of matching and non-matching pairs in the provided dataset.

    :param dataset: the dataset where to count matching and non-matching pairs
    :param label_col: dataset column containing the match/non_match indication
    :param: dictionary containing matching/non_matching number of pairs
    """

    check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(label_col, 'label_col', str, 'string')
    check_cols_in_dataframe(dataset, [label_col])

    match_data = dataset[dataset[label_col] == 1]
    non_match_data = dataset[dataset[label_col] == 0]

    return {'match': len(match_data), 'non_match': len(non_match_data)}