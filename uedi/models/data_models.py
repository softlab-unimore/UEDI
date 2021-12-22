import collections
import pandas as pd
import numpy as np
import random
import operator as op
from functools import reduce
# import py_entitymatching as em


class DataSource(object):
    """
    This class implements the concept of data source: a Pandas dataframe with a unique identifier (i.e., source_id).
    """

    def __init__(self, data, source_id):
        """
        This method initializes a data source.

        :param data: Pandas dataframe
        :param source_id: unique identifier for the data source
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Wrong data type for parameter data. Only Pandas dataframe data type is allowed.")

        if source_id is None:
            raise TypeError("Wrong value for parameter source_id. Provided None value.")

        self.data = data
        self.source_id = source_id
        self.columns = self.data.columns.values
        self.rows = len(self.data)

    def get_data(self):
        """
        This method returns the Pandas dataframe encapsulated inside the DataSource object.
        :return: Pandas dataframe containing the source data
        """
        return self.data

    def get_ids(self):
        """
        This method returns the identifiers of the data source records.
        :return: list containing the identifiers of the data source records
        """
        return self.data.index.values

    def get_num_rows(self):
        """
        This method returns the size of the data source in terms of number of records.
        :return: the size of the data source
        """
        return self.rows

    def get_columns(self):
        """
        This method returns the columns of the data source.
        :return: the column of the data source
        """
        return self.columns

    def get_source_id(self):
        """
        This method returns the unique identifier for the data source.
        :return: the unique identifier of the data source
        """
        return self.source_id

    def filter_data_by_ids(self, ids):
        """
        This method filters the data source records by identifiers.
        :param ids: the identifier of the records to be extracted.
        :return: a new DataSource containing a selected version of the original data
        """
        if not isinstance(ids, collections.Iterable):
            raise TypeError("Wrong data type for parameter indexes. Only iterable data type is allowed.")

        filtered_data = self.data.loc[ids]
        return DataSource(filtered_data, self.source_id)

    def split_randomly(self, seed, num_split=None, sample_ratio=None, debug=False):
        """
        This method randomly splits the data of the data sources.
        The user can specify the split in two ways:
        - by indicating the number of expected splits (num_split parameter): the data will be split in equi-sized
          samples (expect for the last sample, which may contain more elements)
        - by indicating the size ratio through which to split the data (sample_ratio parameter): the data will be split
          in samples based on the user-provided ratio (expect for the last sample, which may contain more elements)
        These two parameters are exclusive: it is possible to set one parameter at a time.

        :param seed: the seed for the random sampling
        :param num_split: number of expected output splits
        :param sample_ratio: data source size ratio through which to split the data
        :param debug: flag for enabling debug mode
        :return: list of DataSource objects
        """

        # check input parameter data types
        if not isinstance(seed, int):
            raise TypeError("Wrong data type for parameter seed. Only integer data type is allowed.")

        if num_split:
            if not isinstance(num_split, int):
                raise TypeError("Wrong data type for parameter num_split. Only integer data type is allowed.")

        if sample_ratio:
            if not isinstance(sample_ratio, float):
                raise TypeError("Wrong data type for parameter sample_ratio. Only integer data type is allowed.")

        # check input parameter data values
        if num_split is None and sample_ratio is None:
            raise ValueError("Provide one of the following parameters: num_split or sample_ratios.")

        if num_split:
            if num_split <= 1 or num_split > self.rows:
                raise ValueError(
                    "Wrong value for parameter num_split. Provide a value included in the range [2, {}]".format(
                        self.rows))

        if sample_ratio:
            if sample_ratio < 0 or sample_ratio >= 1:
                raise ValueError(
                    "Wrong value for the parameter sample_ratio. Only values in the interval [0, 1) are allowed.")

        split_sources = []
        sample_sizes = []

        if num_split:  # split based on num_split parameter

            # split the data in num_split equi-sized samples (expect for the last sample which may contain more
            # elements)
            exact_sample_size = int(self.rows / num_split)
            sample_sizes += [exact_sample_size] * num_split
            if self.rows % num_split != 0:
                sample_sizes[-1] += self.rows - (exact_sample_size * num_split)

            if debug:
                print("Sample sizes: {}".format(sample_sizes))

        else:  # split based on sample_ratio parameter

            # split the data in equi-sized samples based on sample_ratio parameter (expect for the last sample which may
            # contain more elements)
            # count the number of samples that can be extracted from the data source based on the input sample ratio
            num_samples = 0
            cumulative_ratio = sample_ratio
            while cumulative_ratio <= 1:
                cumulative_ratio += sample_ratio
                num_samples += 1

            sample_sizes += [int(self.rows * sample_ratio)] * num_samples

            if self.rows != np.sum(sample_sizes):
                sample_sizes[-1] += self.rows - np.sum(sample_sizes)

            if debug:
                print("Sample sizes: {}".format(sample_sizes))

        # sample the data based on the previuosly computed sample sizes
        remaining_data = self.data.copy()
        for i in range(len(sample_sizes)):
            sample_size = sample_sizes[i]
            sample = remaining_data.sample(n=sample_size, random_state=seed)
            source_data = DataSource(sample, self.source_id)
            split_sources.append(source_data)

            remaining_data = remaining_data.drop(sample.index)

        return split_sources

    def copy(self):
        """
        This method implements the copy of a DataSource object. A new object with a copy of the Pandas dataframe
        will be created.
        :return: a new DataSource, copy of the current object
        """
        return DataSource(self.data.copy(), self.source_id)


class IntegratedDataset(object):
    """
    This class implements the concept of integrated dataset: a dataset obtained by integrating two data sources.
    """

    def __init__(self, data, id_col, source_id_col, entity_label_col):
        """
        This method initialized an integrated dataset.

        :param data: Pandas dataframe containing the data of the integrated dataset
        :param id_col: the name of the column that contains the the identifiers of each record
        :param source_id_col: the name of the column that contains the identifier of the data source of origin for each
               record
        :param entity_label_col:  the name of the column that contains the entity identifier
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Wrong data type for parameter data. Only Pandas dataframe data type is allowed.")

        input_cols = [id_col, source_id_col, entity_label_col]
        cols_name = ["id_col", "source_id_col", "entity_label_col"]
        for i in range(len(input_cols)):
            col = input_cols[i]
            col_name = cols_name[i]
            if not isinstance(col, str):
                raise TypeError("Wrong data type for parameter {}. Only string data type is allowed.".format(col_name))

        cols = data.columns.values
        for i in range(len(input_cols)):
            col = input_cols[i]
            if col not in cols:
                raise TypeError("Column {} not found.".format(col))

        self.data = data
        self.id_col = id_col
        self.source_id_col = source_id_col
        self.entity_label_col = entity_label_col
        self.num_rows = len(data)
        self.source_labels = data[source_id_col].unique()
        self.effectiveness_scores = None

    def set_data(self, data: pd.DataFrame):
        """
        This method sets the data of the integrated dataset.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Wrong data type for parameter data. Only Pandas DataFrame data type is allowed.")

        self.data = data

    def get_data(self):
        """
        This method returns the data of the integrated dataset.
        :return: a Pandas dataframe cotaining the data of the integrated dataset
        """
        return self.data

    def get_id_col(self):
        """
        This method returns the name of the column which stores in the integrated dataset the identifier of each record.
        :return: the name of the column which stores the identifier of each record
        """
        return self.id_col

    def get_source_id_col(self):
        """
        This method returns the name of the column which stores in the integrated dataset the identifier of the data
        source of origin of each record.
        :return: the name of the column which stores the identifier of the data source of origin of each record
        """
        return self.source_id_col

    def get_entity_label_col(self):
        """
        This method returns the name of the column which stores in the integrated dataset the label of the entity
        referred by each record.
        :return: the name of the column which stores the label of the entity referred by each record
        """
        return self.entity_label_col

    def get_effectiveness_scores(self):
        """
        This method returns the effectiveness scores associated to the integrated dataset.
        :return: Python dictionary containing the effectiveness scores
        """

        return self.effectiveness_scores

    def set_effectiveness_scores(self, scores):
        """
        This method sets the effectiveness scores associated to the integrated dataset.
        :return: None
        """

        if not isinstance(scores, dict):
            raise TypeError("Wrong data type for parameter scores. Only Python dictionary data type is allowed.")

        self.effectiveness_scores = scores

    # def compute_effectiveness_scores(self, true_labels):
    #     """
    #     This method compares the entity label column of the current integrated dataset with the user provided true
    #     labels and computes some effectiveness metrics.
    #
    #     :param target_labels: the true labels used to evaluate the effectiveness of the integrated dataset
    #     :return: Python dictionary containing some metrics that evaluate the effectiveness of the integrated dataset
    #     """
    #
    #     if not isinstance(true_labels, collections.Iterable):
    #         raise TypeError("Wrong data type for parameter true_labels. Only iterable data type is allowed.")
    #
    #     predicted_labels = self.data[self.entity_label_col].values
    #
    #     data_for_effectiveness = pd.DataFrame({"pred_label": predicted_labels, "label": true_labels})
    #     eval_result = em.eval_matches(data_for_effectiveness, 'label', 'pred_label')
    #
    #     return eval_result

    def filter_by_source(self, source_val, indexes=None, debug=False):
        """
        This method filters the integrated dataset based on the user-provided data source identifier.
        If the indexes parameter is not set, all the records deriving from source_val data source will be retrieved,
        otherwise only the records with the specified identifiers will be returned.

        :param source_val: the identifier of the data source
        :param indexes: the identifiers of the records of the indicated data source to be retrieved
        :param debug: a flag for enabling the debug mode
        :return: a new IntegratedDataset containing a selected version of the original data
        """

        # check input parameter data types
        if indexes is not None:
            if not isinstance(indexes, collections.Iterable):
                raise TypeError("Wrong data type for parameter indexes. Only iterable data type is allowed.")

        # check input parameter data values
        if source_val is None:
            raise ValueError("Provided None value for parameter source_val.")

        selected_data = self.data[self.data[self.source_id_col] == source_val]

        if indexes is not None:
            selected_data = selected_data[selected_data[self.id_col].isin(indexes)]

        if debug:
            print(selected_data.shape)
            print("Num. elements from the source {}: {}".format(source_val, len(
                selected_data[selected_data[self.source_id_col] == source_val])))
            print("Num. elements NOT from the source: {}".format(
                len(selected_data[selected_data[self.source_id_col] != source_val])))

        selected_integrated = IntegratedDataset(selected_data, self.id_col, self.source_id_col,
                                                self.entity_label_col)

        return selected_integrated

    def filter_by_two_sources(self, source1, source2, debug=False):
        """
        This method filters the integrated dataset based on two data sources.
        All the records belonging to both the data sources will be retrieved.

        :param source1: DataSource 1
        :param source2: DataSource 2
        :param debug: flag for enabling debug mode
        :return: a new IntegratedDataset containing a selected version of the original data
        """

        if not isinstance(source1, DataSource):
            raise TypeError("Wrong data type for parameter source1. Only DataSource data type is allowed.")

        if not isinstance(source2, DataSource):
            raise TypeError("Wrong data type for parameter source2. Only DataSource data type is allowed.")

        extracted_source1_data = self.filter_by_source(source1.get_source_id(), indexes=source1.get_ids(), debug=debug)
        extracted_source2_data = self.filter_by_source(source2.get_source_id(), indexes=source2.get_ids(), debug=debug)
        selected_data = pd.concat([extracted_source1_data.get_data(), extracted_source2_data.get_data()])

        if debug:
            print(selected_data.shape)
            print("Num. of elements from the first data source: {}".format(
                selected_data[selected_data[self.source_id_col] == source1.get_source_id()].shape))
            print("Num. of elements from the second data source: {}".format(
                selected_data[selected_data[self.source_id_col] == source2.get_source_id()].shape))

        selected_integrated = IntegratedDataset(selected_data, self.id_col, self.source_id_col,
                                                self.entity_label_col)

        return selected_integrated

    def filter_by_index(self, indexes):
        """
        This method filters the integrated data records by identifiers.
        :param indexes: the identifier of the records to be extracted.
        :return: a new IntegratedDataset containing a selected version of the original data
        """

        if not isinstance(indexes, collections.Iterable):
            raise TypeError("Wrong data type for parameter indexes. Only iterable data type is allowed.")

        selected_data = self.data.loc[indexes]
        selected_integrated = IntegratedDataset(selected_data, self.id_col, self.source_id_col,
                                                self.entity_label_col)

        return selected_integrated

    def flip_randomly_entity_id(self, seed, num_flip=None, error_rate=None, support=1, debug=True):
        """
        This method transforms the integrated data by flipping some identifier values. The user can indicates the
        absolute number of flips to be performed or an error_rate which indicates the number of flips with respect the
        size of the integrated dataset (or its selection based on the support parameter). If specified, the
        method flips only the values of the entities with a number of records greater than the support parameter.

        :param seed: seed for the random choices
        :param num_flip: number of flip to be performed
        :param error_rate: error rate to be injected in the data
        :param support: minimum number of records included in an entity
        :param debug: flag for enabling debug mode
        :return: (transformed_integration, sampled_indexes, new_values)
            - transformed_integration: the new IntegratedDataset
            - sampled_indexes: the indexes of the flipped records
            - new values: the modified entity identifiers
        """

        # check data types
        if not isinstance(seed, int):
            raise TypeError("Wrong data type for parameter seed. Only integer data type is allowed.")

        if num_flip is not None:
            try:
                num_flip = int(num_flip)
            except Exception:
                raise TypeError("Wrong data type for parameter num_flip. Only numeric data type is allowed.")

        if error_rate is not None:
            try:
                error_rate = float(error_rate)
            except Exception:
                raise TypeError("Wrong data type for parameter error_rate. Only numeric data type is allowed.")

        try:
            support = int(support)
        except Exception:
            raise TypeError("Wrong data type for parameter support. Only numeric data type is allowed.")

        # check data values
        if num_flip is not None and error_rate is not None:
            raise ValueError("Provide only one of the following parameters: num_flip, error_rate (not both).")

        if num_flip is not None:
            if num_flip <= 0 or num_flip > self.num_rows:
                raise ValueError(
                    "Parameter num_flip out of range. Only values in the range [1, {}] are allowed.".format(
                        self.num_rows))

        if error_rate is not None:
            if error_rate <= 0 or error_rate >= 1:
                raise ValueError(
                    "Parameter error_rate out of range. Only values in the range (0, 1) are allowed.".format(
                        self.num_rows))

        unique_counts = self.data[self.entity_label_col].value_counts()
        max_support = unique_counts.max()
        if support <= 0 or support > max_support:
            raise ValueError("Parameter support out of range. Only values in the range [1, {}] are allowed.".format(
                max_support))

        if support > 1 and num_flip is not None:
            unique_counts_gt_support = unique_counts[unique_counts >= support]
            if len(unique_counts_gt_support) < num_flip:
                error_msg = "No enough elements to be sampled with support {}.".format(support)
                error_msg += " Available {} elements, but required {} elements.".format(len(unique_counts_gt_support),
                                                                                        num_flip)
                raise ValueError(error_msg)

        unique_values = self.data[self.entity_label_col].unique()

        selected_df = self.data[
            self.data.groupby(self.entity_label_col)[self.entity_label_col].transform('size') >= support]

        if error_rate is not None:
            num_flip = int(error_rate * len(selected_df))

        sampled_indexes = random.choices(selected_df.index.values, k=num_flip)

        if debug:
            print(self.data.loc[sampled_indexes])

        random.seed(seed)

        new_column = self.data[self.entity_label_col].copy()
        new_values = []
        for sampled_index in sampled_indexes:
            current_val = new_column.loc[sampled_index]
            remaining_unique_values = list(set(unique_values).difference(set([current_val])))
            new_value = random.choice(remaining_unique_values)
            new_column.loc[sampled_index] = new_value
            new_values.append(new_value)

        transformed_df = self.data.copy()
        transformed_df[self.entity_label_col] = new_column
        transformed_integration = IntegratedDataset(transformed_df, self.id_col, self.source_id_col,
                                                    self.entity_label_col)

        return transformed_integration, sampled_indexes, new_values

    def copy(self):
        """
        This method implements the copy of an IntegratedDataset object. A new object with a copy of the Pandas dataframe
        will be created.
        :return: a new IntegratedDataset, copy of the current object
        """
        return IntegratedDataset(self.data.copy(), self.id_col, self.source_id_col, self.entity_label_col)

    def describe(self):
        """
        This method provides a description of the integrated dataset.
        :return: None
        """

        # num records
        num_records = len(self.data)
        print("Num. records: {}\n".format(num_records))

        # num entities
        num_entities = len(self.data[self.entity_label_col].unique())
        print("Num. entities: {}\n".format(num_entities))

        # num records per entity
        print("Entities grouped by records")
        num_records_per_entity = self.data[self.entity_label_col].value_counts()
        entities_grouped_by_records = num_records_per_entity.value_counts()
        print("{}\n".format(entities_grouped_by_records))

        # matches
        def ncr(n, r):  # number of combinations
            r = min(r, n - r)
            numer = reduce(op.mul, range(n, n - r, -1), 1)
            denom = reduce(op.mul, range(1, r + 1), 1)
            return numer / denom

        num_matches = 0
        for rec_per_ent, freq_ent_grouped_by_rec in entities_grouped_by_records.items():
            if rec_per_ent != 1:
                num_matches += ncr(rec_per_ent, 2) * freq_ent_grouped_by_rec
        match_percentage = (num_matches / num_entities) * 100
        print("Match percentage: {} / {} = {}\n".format(num_matches, num_entities, match_percentage))

        # num sources
        sources = self.data[self.source_id_col].unique()
        print("Sources: {} ({})\n".format(sources, len(sources)))

        # records by source
        num_records_by_source = self.data[self.source_id_col].value_counts()
        print("Num. records by source")
        print("{}\n".format(num_records_by_source))

        # matches from sources
        records_in_matched_entities = self.data.groupby(self.entity_label_col).filter(lambda x: len(x) > 1)
        records_in_matched_entities_by_source = records_in_matched_entities.groupby(self.source_id_col).size()
        print("Matches by source")
        print("{}\n".format(records_in_matched_entities_by_source))

    def generate_datasets_for_data_integration(self, data_type="all", same_size=False):
        """
        This function produces a series of datasets by splitting the integrated data in different modalities.

        :param data_type: string parameter that indicates which typology of data is used to create the output datasets
                          (three choices are allowed: all data, match data, non-match data)
        :param same_size: boolean parameter that specifies if the output data sources have to be the same size or not
        :return: (list of data sources, list of integrated data)
        """

        if not isinstance(data_type, str):
            raise TypeError("Wrong data type for parameter data_type. Only string data type is allowed.")

        if not isinstance(same_size, bool):
            raise TypeError("Wrong data type for parameter same_size. Only boolean data type is allowed.")

        allowed_data_types = ["all", "match", "non_match"]
        if data_type not in allowed_data_types:
            raise ValueError("Wrong data value for parameter data_type. Only values in the list {} are allowed.".format(
                allowed_data_types))

        data_sources = []
        integrations = []

        if data_type == "all":

            if not same_size:

                # get data sources from the integrated dataset
                for source in self.source_labels:
                    s_data = self.data[self.data[self.source_id_col] == source]
                    data_source = DataSource(s_data, source)
                    data_sources.append(data_source)

                integrations.append(self.copy())

            else:

                # get the size of the data source with the minimum number of rows in the integrated dataset
                min_size = self.data.groupby(self.source_id_col).size().min()

                # get data sources from the integrated dataset and select the first min_size rows
                for source in self.source_labels:
                    select_data = self.data[self.data[self.source_id_col] == source].iloc[:min_size, :]
                    select_data_source = DataSource(select_data, source)
                    data_sources.append(select_data_source)

                # concat data sources in order to build the new integration dataset
                integration_data = pd.concat([data.get_data() for data in data_sources])
                same_size_integration = IntegratedDataset(integration_data, self.id_col, self.source_id_col,
                                                          self.entity_label_col)
                integrations.append(same_size_integration)

                # check if data has the same size
                data_sizes = [len(source.get_data()) for source in data_sources]
                assert len(np.unique(data_sizes)) == 1

        elif data_type == "match":

            # retrieve from the integrated dataset only the rows that refer to matches
            integration_match_data = self.data.groupby(self.entity_label_col).filter(lambda x: len(x) > 1)

            if not same_size:

                # get data sources from the matches of the integrated dataset
                for source in self.source_labels:
                    data_match = integration_match_data[integration_match_data[self.source_id_col] == source]
                    data_source_match = DataSource(data_match, source)
                    data_sources.append(data_source_match)

                # emit as integrated dataset the matches rows
                integration_match = IntegratedDataset(integration_match_data, self.id_col, self.source_id_col,
                                                      self.entity_label_col)
                integrations.append(integration_match)

            else:

                # get the size of the data source with the minimum number of rows in the matches of the integrated
                # dataset
                match_min_size = integration_match_data.groupby(self.source_id_col).size().min()

                # get data sources from the integrated dataset and select the first match_min_size rows
                for source in self.source_labels:
                    match_select_data = integration_match_data[
                                            integration_match_data[self.source_id_col] == source].iloc[
                                        :match_min_size, :]
                    match_select_data_source = DataSource(match_select_data, source)
                    data_sources.append(match_select_data_source)

                # concat data sources in order to build the new integration dataset
                integration_data = pd.concat([data.get_data() for data in data_sources])
                same_size_match_integration = IntegratedDataset(integration_data, self.id_col, self.source_id_col,
                                                                self.entity_label_col)
                integrations.append(same_size_match_integration)

                # check if data has the same size
                data_sizes = [len(source.get_data()) for source in data_sources]
                assert len(np.unique(data_sizes)) == 1

                # check if integration data has a size equals to the sum of the data source sizes
                assert len(integration_data) == np.sum(data_sizes)

                # check if integration data contains a number of data sources equals to the number of total data sources
                assert len(integration_data[self.source_id_col].unique()) == len(data_sources)

            # check if data contains only matches by searching in the data sources if there is an entity id
            # (among the entity ids included in the original integrated dataset) that is not included in the matched
            # entity ids
            all_entity_ids = self.data[self.entity_label_col].unique()
            match_entity_ids = integration_match_data[self.entity_label_col].unique()
            non_match_entity_ids = set(all_entity_ids).difference(set(match_entity_ids))
            for data_source in data_sources:
                source = data_source.get_data()
                assert len(source[source[self.entity_label_col].isin(non_match_entity_ids)]) == 0

        elif data_type == "non_match":

            # retrieve from the integrated dataset only the rows that refer to non-match
            integration_non_match_data = self.data.groupby(self.entity_label_col).filter(lambda x: len(x) == 1)

            if not same_size:

                # get data sources from the non-matches of the integrated dataset
                for source in self.source_labels:
                    data_non_match = integration_non_match_data[
                        integration_non_match_data[self.source_id_col] == source]
                    data_source_non_match = DataSource(data_non_match, source)
                    data_sources.append(data_source_non_match)

                # emit as integrated dataset the non-matches rows
                integration_non_match = IntegratedDataset(integration_non_match_data, self.id_col, self.source_id_col,
                                                          self.entity_label_col)
                integrations.append(integration_non_match)

            else:

                # get the size of the data source with the minimum number of rows in the non-matches of the integrated
                # dataset
                non_match_min_size = integration_non_match_data.groupby(self.source_id_col).size().min()
                for source in self.source_labels:
                    non_match_select_data = integration_non_match_data[
                                                integration_non_match_data[self.source_id_col] == source].iloc[
                                            :non_match_min_size, :]
                    non_match_select_data_source = DataSource(non_match_select_data, source)
                    data_sources.append(non_match_select_data_source)

                # concat data sources in order to build the new integration dataset
                integration_data = pd.concat([data.get_data() for data in data_sources])
                same_size_non_match_integration = IntegratedDataset(integration_data, self.id_col, self.source_id_col,
                                                                    self.entity_label_col)
                integrations.append(same_size_non_match_integration)

                # check if data has the same size
                data_sizes = [len(source.get_data()) for source in data_sources]
                print(data_sizes)
                assert len(np.unique(data_sizes)) == 1

            # check if data contains only non-matches by searching in the data sources if there is an entity id
            # (among the entity ids included in the original integrated dataset) that is included in the matched
            # entity ids
            all_entity_ids = self.data[self.entity_label_col].unique()
            non_match_entity_ids = integration_non_match_data[self.entity_label_col].unique()
            non_match_entity_ids = set(all_entity_ids).difference(set(non_match_entity_ids))
            for data_source in data_sources:
                source = data_source.get_data()
                assert len(source[source[self.entity_label_col].isin(non_match_entity_ids)]) == 0

        return data_sources, integrations
