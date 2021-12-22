import pandas as pd
import networkx as nx
import os
import logging
from uedi.models.data_models import DataSource, IntegratedDataset
from uedi.utils.general_utilities import check_parameter_type, check_cols_in_dataframe


class IntegratedDataPreparationComponent(object):
    """
    This class manages the preparation of an integrated data source.
    """

    def __init__(self, matching_pairs_data, match_id_col, left_id_col, right_id_col, match_label_col, out_entity_col):
        """
        This method initializes the main variables related to an integrated data source.
        The main input of this method is a file containing some labelled record matching pairs.
        The expected data format of this file is the following:
            <matching_pair_id> <matching_pair_left_id> <matching_pair_right_id> <matching_pair_label>
        where <matching_pair_left_id> (<matching_pair_right_id>) is the identifier of the record in the matching pair
        that derives from the SAME left (right) data source.
        In other words, like in the Magellan data format, two assumptions are made:
        - the matching pairs come from only two data sources
        - the records of the first data source are located in the left side of the matching pairs and that from the
        second data source are in the right side.
        The file can contain also other information (e.g., Magellan stores all the attributes that describe the left and
        the right records, not only their identifiers), but it will be ignored.

        :param matching_pairs_data: Pandas DataFrame object containing the matching pairs
        :param match_id_col: the column name that contains the identifier of each matching pair
        :param left_id_col: the column name that contains the identifier of the left record of each matching pair
        :param right_id_col: the column name that contains the identifier of the right record of each matching pair
        :param out_entity_col: the column name that contains the identifier of the output entities
        :param match_label_col: the column name that contains the label (match/non-match) of each matching pair
        """

        # check input parameter data types
        check_parameter_type(matching_pairs_data, 'matching_pairs_data', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(match_id_col, 'match_id_col', str, 'string')
        check_parameter_type(left_id_col, 'left_id_col', str, 'string')
        check_parameter_type(right_id_col, 'right_id_col', str, 'string')
        check_parameter_type(match_label_col, 'match_label_col', str, 'string')

        self.complete_matching_pairs = matching_pairs_data
        user_provided_columns = [match_id_col, left_id_col, right_id_col, match_label_col]
        check_cols_in_dataframe(self.complete_matching_pairs, user_provided_columns)

        self.match_id_col = match_id_col
        self.left_id_col = left_id_col
        self.right_id_col = right_id_col
        self.match_label_col = match_label_col
        self.entity_id_col = out_entity_col

        # if the file contains other information besides the input-provided column names, ignore it
        # this is the case of Magellan which stores all the attributes that describe the left and the right records
        # of the matching pairs
        self.matching_pairs = self.complete_matching_pairs[[left_id_col, right_id_col, match_label_col]]
        self.matching_pairs.index = self.complete_matching_pairs[match_id_col]
        print("\nINTEGRATED DATA SOURCE: {}".format(self.matching_pairs.shape))

        # num matching pairs
        self.num_matching_pairs = self.matching_pairs.shape[0]
        print("\tNum matching pairs: {}".format(self.num_matching_pairs))

        # matches and non-matches
        self.matches = self.matching_pairs[self.matching_pairs[match_label_col] == 1][[left_id_col, right_id_col]]
        self.num_matches = self.matches.shape[0]
        matches_percent = float("{0:.2f}".format((self.num_matches / float(self.num_matching_pairs)) * 100))
        print("\tPercentage of matches: {} / {} = {}".format(self.num_matches, self.num_matching_pairs, matches_percent))

        self.non_matches = self.matching_pairs[self.matching_pairs[match_label_col] == 0][[left_id_col, right_id_col]]
        self.num_non_matches = self.non_matches.shape[0]

        # num entities
        self.left = self.matching_pairs[left_id_col].unique()
        self.right = self.matching_pairs[right_id_col].unique()
        self.num_entities = len(self.left) + len(self.right)
        print("\tNumber of entities: {}".format(self.num_entities))

        self.cluster_entities = None
        self.integrated_data = None

    def _cluster_record_ids_by_entity(self):
        """
        This method transitively extends the matching pairs between records in order to obtain clusters of records that
        refer to the same real-world entity.
        This method organizes the matching pairs with a graph data structure and then performs their transitive
        extension by applying a standard connected component algorithm.
        :return: a list of sets, where each set contains the record ids that refer to the same real-world entity
        """

        # STEP 1: creation of the graph of matches

        # organize ONLY THE MATCHES in a graph data structure
        # get unique left and right matches record ids and prefix them with the "name" of the data source of origin
        # these extended ids are the names of the nodes of the graph
        left_matches = ["left_{}".format(l) for l in self.matches[self.left_id_col].unique()]
        right_matches = ["right_{}".format(r) for r in self.matches[self.right_id_col].unique()]
        unique_matches = left_matches + right_matches

        # create the graph of matches
        matches_graph = nx.Graph()
        for node in unique_matches:  # nodes
            matches_graph.add_node(node)
        for match_pair in self.matches.values:  # edges
            matches_graph.add_edge("left_{}".format(match_pair[0]), "right_{}".format(match_pair[1]))

        # plot graph of matches
        # fig = plt.figure(figsize=(20, 10))
        # edges = matches_graph.edges()
        # pos = nx.spring_layout(matches_graph)
        # nx.draw_networkx_nodes(matches_graph, pos, node_size = 200)
        # nx.draw_networkx_labels(matches_graph, pos)
        # nx.draw_networkx_edges(matches_graph, pos, edgelist=edges, arrows=False)

        # STEP 2: find clusters of records by entity

        # apply connected components algorithm
        cluster_entities = []
        ccs = nx.connected_components(matches_graph)
        for cc in ccs:
            cluster_entities.append(cc)

        # add in the final result also the records that haven't matched with any other record
        # some records in non-matches pairs can also appear in matches pairs
        # remove them from the list of the non_matches records
        left_non_matches = ["left_{}".format(l) for l in self.non_matches[self.left_id_col].unique()]
        right_non_matches = ["right_{}".format(r) for r in self.non_matches[self.right_id_col].unique()]
        left_exclusive_non_matches = set(left_non_matches) - set(left_matches)
        right_exclusive_non_matches = set(right_non_matches) - set(right_matches)
        cluster_entities.extend([{l} for l in left_exclusive_non_matches])
        cluster_entities.extend([{l} for l in right_exclusive_non_matches])

        return cluster_entities

    def _convert_clusters_to_tabular(self, data1, data2, data1_id, data2_id, source_col, data1_index_col=None,
                                     data2_index_col=None):
        """
        This method converts a list of sets, where each set contains the record ids that refer to the same real-world
        entity, in a tabular format.
        This method takes in input two data sources which are exploited to access to integral information of a record
        starting from its identifier.
        To preserve the information of the real-world entity referred by each record, a new column 'entity_id' is added
        to the final tabular format.

        :param data1: Pandas DataFrame object containing the first dataset
        :param data2: Pandas DataFrame object containing the first dataset
        :param data1_id: identifier of the first dataset
        :param data2_id: identifier of the second dataset
        :param source_col: name of the new column that will store the data provenance information
        :param data1_index_col: the primary key column in the first dataset
        :param data2_index_col: the primary key column in the second dataset
        :return: Pandas DataFrame object that contains records labelled by entity id
        """

        check_parameter_type(data1, 'data1', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(data2, 'data2', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(data1_id, 'data1_id', int, 'integer')
        check_parameter_type(data2_id, 'data2_id', int, 'integer')
        check_parameter_type(source_col, 'source_col', str, 'string')
        check_parameter_type(data1_index_col, 'data1_index_col', str, 'string', optional_param=True)
        check_parameter_type(data2_index_col, 'data2_index_col', str, 'string', optional_param=True)

        if data1_index_col is not None:
            check_cols_in_dataframe(data1, [data1_index_col])

        if data2_index_col is not None:
            check_cols_in_dataframe(data2, [data2_index_col])

        if self.cluster_entities is None:
            raise Exception("This method has to be executed only after the method 'cluster_record_ids_by_entity'.")

        entity_records = []
        entity_id = 1

        data1_with_index = data1.reset_index()
        data2_with_index = data2.reset_index()

        # loop over clusters of record ids
        for cluster in self.cluster_entities:

            # save record ids that refer to left (right) data source (i.e., data1/data2)
            left_ids = []
            right_ids = []
            for record_id in cluster:
                if record_id.startswith("left"):
                    alphanumeric_record_id = record_id.replace("left_", "")
                    try:
                        alphanumeric_record_id = int(alphanumeric_record_id)
                    except ValueError:
                        pass
                    left_ids.append(alphanumeric_record_id)
                else:
                    alphanumeric_record_id = record_id.replace("right_", "")
                    try:
                        alphanumeric_record_id = int(alphanumeric_record_id)
                    except ValueError:
                        pass
                    right_ids.append(alphanumeric_record_id)

            # retrieve the records original content based on record ids
            if data1_index_col is None:
                for l_id in left_ids:
                    if not isinstance(l_id, int):
                        raise TypeError("Wrong data type found in left_ids list. Found non integer value in the iloc mode.")
                left_records = data1_with_index.iloc[left_ids, :].copy()
            else:
                left_records = data1_with_index[data1_with_index[data1_index_col].isin(left_ids)].copy()
            if len(left_records) == 0 and len(left_ids) > 0:
                logging.warning("No records found in first source with ids: {}".format(left_ids))

            if data2_index_col is None:
                for r_id in right_ids:
                    if not isinstance(r_id, int):
                        raise TypeError("Wrong data type found in right_ids list. Found non integer value in the iloc mode.")
                right_records = data2_with_index.iloc[right_ids, :].copy()
            else:
                right_records = data2_with_index[data2_with_index[data2_index_col].isin(right_ids)].copy()
            if len(right_records) == 0 and len(right_ids) > 0:
                logging.warning("No records found in second source with ids: {}".format(right_ids))

            left_records[source_col] = data1_id
            right_records[source_col] = data2_id

            # concat left and right records into a unique table and assign them the same
            # entity id
            records_same_entity = pd.concat([left_records, right_records])
            records_same_entity[self.entity_id_col] = entity_id

            entity_id += 1
            entity_records.append(records_same_entity)

        # create a final table that contains all records labeled by entity id
        entity_records_table = pd.concat(entity_records)
        entity_records_table = entity_records_table.reset_index(drop=True)

        return entity_records_table

    def group_records_by_entity(self, data1, data2, data1_id, data2_id, source_col, data1_index_col=None,
                                data2_index_col=None):
        """
        This method performs two task:
        1. transitively extends the matching pairs between records in order to obtain clusters of records that
        refer to the same real-world entity
        2. converts a list of sets, where each set contains the record ids that refer to the same real-world
        entity, in a tabular format.

        :param data1: Pandas DataFrame object containing the first dataset
        :param data2: Pandas DataFrame object containing the first dataset
        :param data1_id: identifier of the first dataset
        :param data2_id: identifier of the second dataset
        :param source_col: name of the new column that will store the data provenance information
        :param data1_index_col: the primary key column in the first dataset
        :param data2_index_col: the primary key column in the second dataset
        :return: Pandas dataframe object that contains records labelled by entity id
        """

        self.cluster_entities = self._cluster_record_ids_by_entity()
        self.integrated_data = self._convert_clusters_to_tabular(data1, data2, data1_id, data2_id, source_col,
                                                                 data1_index_col=data1_index_col,
                                                                 data2_index_col=data2_index_col)

        return self.integrated_data

    def get_left_entities(self):
        """
        This method returns the identifiers of the records which come from the left data source.
        :return: left data source record ids
        """
        return self.left

    def get_right_entities(self):
        """
        This method returns the identifiers of the records which come from the right data source.
        :return: right data source record ids
        """
        return self.right

    def get_matching_pairs(self):
        """
        This method returns, in a Pandas dataframe object, the labelled matching pairs. The adopted data format is the
        following:
        - dataframe index: <matching_pair_id>
        - dataframe columns: <matching_pair_left_id> <matching_pair_right_id> <matching_pair_label>
        :return: Pandas dataframe object containing the labelled matching pairs
        """
        return self.matching_pairs

    def get_num_matching_pairs(self):
        """
        This method returns the number of labelled matching pairs.
        :return: number of labelled matching pairs
        """
        return self.num_matching_pairs

    def get_matches(self):
        """
        This method returns the matching pairs labelled as match.
        :return: the matching pairs labelled as match
        """
        return self.matches

    def get_num_matches(self):
        """
        This method returns the number of matching pairs labelled as match.
        :return: the number of matching pairs labelled as match
        """
        return self.num_matches

    def get_non_matches(self):
        """
        This method returns the matching pairs labelled as non-match.
        :return: the matching pairs labelled as non-match
        """
        return self.non_matches

    def get_num_non_matches(self):
        """
        This method returns the number of matching pairs labelled as non-match.
        :return: the number of matching pairs labelled as non-match
        """
        return self.num_non_matches

    def get_cluster_entities(self):
        """
        This method returns a list of sets, where each set contains the record ids that refer to the same real-world
        entity.
        :return: a list of sets, where each set contains the record ids that refer to the same real-world entity
        """
        return self.cluster_entities

    def get_integrated_data(self):
        """
        This method returns an unmerged tabular version of the matching pairs dataset.
        :return: a Pandas dataframe object containing an unmerged tabular version of the matching pairs dataset
        """
        return self.integrated_data


def convert_matching_pairs_to_integrated_dataset(data1, data1_id, data1_index_col, data2, data2_id, data2_index_col,
                                                 match_pairs, match_id, left_id, right_id, match_label):
    """
    This function converts a list of matching pairs into an integrated dataset.

    :param data1: Pandas DataFrame object containing the first dataset
    :param data1_id: identifier of the first dataset
    :param data1_index_col: the primary key column in the first dataset
    :param data2: Pandas DataFrame object containing the first dataset
    :param data2_id: identifier of the second dataset
    :param data2_index_col: the primary key column in the second dataset
    :param match_pairs: Pandas DataFrame object containing the matching pairs
    :param match_id: the primary key column in in the matching pairs data
    :param left_id: the foreign key column in the matching pairs data that refers to the records of the first dataset
    :param right_id: the foreign key column in the matching pairs data that refers to the records of the second dataset
    :param match_label: the column in the matching pairs data that identifies as match or non-match each matching pair
    :return: first dataset, second dataset, integrated dataset
    """

    # check input parameter data types
    check_parameter_type(data1, 'data1', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(data1_id, 'data1_id', int, 'integer')
    check_parameter_type(data1_index_col, 'data1_index_col', str, 'string')
    check_parameter_type(data2, 'data2', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(data2_id, 'data2_id', int, 'integer')
    check_parameter_type(data2_index_col, 'data2_index_col', str, 'string')
    check_parameter_type(match_pairs, 'match_pairs', pd.DataFrame, 'Pandas DataFrame')
    check_parameter_type(match_id, 'match_id', str, 'string')
    check_parameter_type(left_id, 'left_id', str, 'string')
    check_parameter_type(right_id, 'right_id', str, 'string')
    check_parameter_type(match_label, 'match_label', str, 'string')

    # check input parameter data values
    check_cols_in_dataframe(data1, [data1_index_col])
    check_cols_in_dataframe(data2, [data2_index_col])
    match_pairs_user_columns = [match_id, left_id, right_id, match_label]
    check_cols_in_dataframe(match_pairs, match_pairs_user_columns)

    source_col = "source"
    out_entity_id_col = "entity_id"
    integrated_preparation_component = IntegratedDataPreparationComponent(match_pairs, match_id, left_id, right_id,
                                                                          match_label, out_entity_id_col)
    integrated_data_content = integrated_preparation_component.group_records_by_entity(data1, data2, data1_id,
                                                                                       data2_id, source_col,
                                                                                       data1_index_col=data1_index_col,
                                                                                       data2_index_col=data2_index_col)

    # the integrated dataset may contain a subset of the entities included in the orginal data sources
    # filter out from original data sources the entities not included in the integrated dataset

    # get left and right entities included in the integrated dataset
    left_entities = integrated_preparation_component.get_left_entities()
    right_entities = integrated_preparation_component.get_right_entities()

    # filter original data sources
    data1_content = data1[data1[data1_index_col].isin(left_entities)]
    data2_content = data2[data2[data2_index_col].isin(right_entities)]

    # create final data sources and their integrated version
    data1 = DataSource(data1_content, data1_id)
    data2 = DataSource(data2_content, data2_id)
    integrated_data = IntegratedDataset(integrated_data_content, "index", source_col, out_entity_id_col)

    return data1, data2, integrated_data


class DataPreparationComponent(object):
    """
    This class manages the preparation of data (data sources and integrated data).
    """

    def __init__(self, data1_file, data1_id, data2_file, data2_id, integrated_data_file, match_id, left_id, right_id,
                 match_label, data1_index_col=None, data2_index_col=None):
        """
        This method prepares data sources and integrated data based on the user-provided data files.
        This method applies two main tasks:
        - create an integrated dataset
        - filter out from the original data sources the entities that are not included in the integrated dataset

        :param data1_file: file of the first data source
        :param data1_id: identifier of the first data source
        :param data2_file: file of the second data source
        :param data2_id: identifier of the second data source
        :param integrated_data_file: file of the integrated data
        :param match_id: the primary key column in in the integrated data
        :param left_id: the foreign key column in the integrated data that refers to the records of the first dataset
        :param right_id: the foreign key column in the integrated data that refers to the records of the second dataset
        :param match_label: the column in the integrated data that identifies as match or non-match each matching pair
        :param data1_index_col: the primary key column in the first dataset
        :param data2_index_col: the primary key column in the second dataset
        """

        # check input parameter data types
        input_params_map = {"data1": (data1_file, str, "string")}
        input_params_map["data1_id"] = (data1_id, int, "integer")
        input_params_map["data2_file"] = (data2_file, str, "string")
        input_params_map["data2_id"] = (data2_id, int, "integer")
        input_params_map["integrated_data_file"] = (integrated_data_file, str, "string")
        input_params_map["match_id"] = (match_id, str, "string")
        input_params_map["left_id"] = (left_id, str, "string")
        input_params_map["right_id"] = (right_id, str, "string")
        input_params_map["match_label"] = (match_label, str, "string")
        input_params_map["data1_index_col"] = (data1_index_col, str, "string")
        input_params_map["data2_index_col"] = (data2_index_col, str, "string")

        for input_param_name in input_params_map:
            input_param = input_params_map[input_param_name]
            input_param_value = input_param[0]
            input_param_expected_type = input_param[1]
            msg_data_type = input_param[2]

            if input_param_value is not None:

                if not isinstance(input_param_value, input_param_expected_type):
                    raise TypeError(
                        "Wrong data type for parameter {}. Only {} data type is allowed.".format(input_param_name,
                                                                                                 msg_data_type))

        input_files = [data1_file, data2_file, integrated_data_file]
        for input_file in input_files:
            if not os.path.exists(input_file):
                raise ValueError("File {} doesn't exist.".format(input_file))

        # data source 1
        print("********** ORIGINAL DATA **********")
        original_data1 = pd.read_csv(data1_file)
        print("\nORIGINAL DATA SOURCE1: {}".format(original_data1.shape))

        # data source 2
        original_data2 = pd.read_csv(data2_file)
        print("\nORIGINAL DATA SOURCE2: {}".format(original_data2.shape))

        # integrated data source
        # Magellan format contains in the first 5 rows some metadata. let's ignore them
        original_integrated_data = pd.read_csv(integrated_data_file, skiprows=5)

        # check input parameter data values
        match_pairs_user_columns = [match_id, left_id, right_id, match_label]
        match_pairs_user_columns_names = ["match_id", "left_id", "right_id", "match_label"]
        for col, col_name in zip(match_pairs_user_columns, match_pairs_user_columns_names):
            if col not in original_integrated_data.columns.values:
                raise ValueError(
                    "Wrong data value for parameter {}. Column {} not found in the integrated data.".format(col_name,
                                                                                                              col))

        if data1_index_col is not None:
            if data1_index_col not in original_data1.columns.values:
                raise ValueError("Wrong data value for parameter data1_index_col. Column not found in DataFrame data1.")

        if data2_index_col is not None:
            if data2_index_col not in original_data2.columns.values:
                raise ValueError("Wrong data value for parameter data2_index_col. Column not found in DataFrame data2.")

        self.data1_file = data1_file
        self.data2_file = data2_file
        self.integrated_data_file = integrated_data_file

        source_col = "source"
        out_entity_id_col = "entity_id"
        integrated_preparation_component = IntegratedDataPreparationComponent(original_integrated_data, match_id,
                                                                              left_id, right_id, match_label,
                                                                              out_entity_id_col)

        integrated_data_content = integrated_preparation_component.group_records_by_entity(original_data1,
                                                                                           original_data2, data1_id,
                                                                                           data2_id, source_col,
                                                                                           data1_index_col = data1_index_col,
                                                                                           data2_index_col = data2_index_col)


        print("\n***********************************\n")

        # the integrated dataset may contain a subset of the entities included in the orginal data sources
        # filter out from original data sources the entities not included in the integrated dataset

        # get left and right entities included in the integrated dataset
        left_entities = integrated_preparation_component.get_left_entities()
        right_entities = integrated_preparation_component.get_right_entities()

        # filter original data sources
        if data1_index_col is not None:
            data1_content = original_data1[original_data1[data1_index_col].isin(left_entities)]
        else:
            data1_content = original_data1.loc[left_entities]

        if data2_index_col is not None:
            data2_content = original_data2[original_data2[data2_index_col].isin(right_entities)]
        else:
            data2_content = original_data2.loc[right_entities]

        # create final data sources and their integrated version
        print(" ****** DATA AFTER SELECTION ******")
        self.data1 = DataSource(data1_content, data1_id)
        self.data2 = DataSource(data2_content, data2_id)
        self.integrated_data = IntegratedDataset(integrated_data_content, "index", source_col, out_entity_id_col)
        print("\nSOURCE1: {}".format(data1_content.shape))
        print("\nSOURCE2: {}".format(data2_content.shape))
        print("\nINTEGRATION: {}".format(integrated_data_content.shape))
        print("\n***********************************\n")

    def get_all_data(self):
        """
        This method returns the data sources and their integrated version.

        :return: (list of data sources, list of integrated datasets)
        """
        return [self.data1, self.data2], [self.integrated_data]

    def save_all_data(self, out_dir):
        """
        This method saves the prepared data into disk.

        :param out_dir: directory where to save the prepared datasets
        :return: None
        """

        if not isinstance(out_dir, str):
            raise TypeError("Wrong data type for parameter out_dir. Only string data type is allowed.")

        if not os.path.isdir(out_dir):
            raise ValueError("Directory {} doesn't exist.".format(out_dir))

        data1_name = self.data1_file.split(os.sep)[-1]
        out_data1_name = os.path.join(out_dir, data1_name)
        data2_name = self.data2_file.split(os.sep)[-1]
        out_data2_name = os.path.join(out_dir, data2_name)
        integration_name = self.integrated_data_file.split(os.sep)[-1]
        out_integration_name = os.path.join(out_dir, integration_name)

        self.data1.get_data().to_csv(out_data1_name, index=False)
        self.data2.get_data().to_csv(out_data2_name, index=False)
        self.integrated_data.get_data().to_csv(out_integration_name, index=False)

    def split_randomly_data_sources(self, num_sources_out, seed, mixed=False, debug=False):
        """
        This method splits randomly the original data sources. Two split techniques are available, which differ in
        mixing (or not) the information of the original data sources.

        :param num_sources_out: number of sources expected in output
        :param seed: seed for random sampling
        :param mixed: flag for selecting the split technique
        :param debug: flag for enabling debug modality
        :return: a list of data sources
        """

        # check input parameter data types
        if not isinstance(num_sources_out, int):
            raise TypeError("Wrong data type for parameter num_sources_out. Only integer data type is allowed.")

        if not isinstance(seed, int):
            raise TypeError("Wrong data type for parameter seed. Only integer data type is allowed.")

        if not isinstance(mixed, bool):
            raise TypeError("Wrong data type for parameter mixed. Only boolean data type is allowed.")

        # check input parameter data values
        if num_sources_out <= 2:
            raise ValueError("Wrong value for parameter num_sources_out. Provide a value grater than 2.")

        # STEP 1: split data sources
        data_sources = []

        # two splitting techniques are available:
        # non-mixed: the data sources are split in num_sources_out samples without mixing their information
        # mixed: the data sources are split in num_sources_out samples mixing their information

        if not mixed:  # no mix

            # the two data sources are split in equi-sized samples without mixing their records
            # if the number of output sources if odd then from the first data source is extracted a sample more than the
            # second one

            # the output data sources are generated (if possible) by sampling the same number of samples from the first
            # and the second original data sources
            same_num_samples = int(num_sources_out / 2)

            if debug:
                print("Num samples: {}".format(same_num_samples))

            # if the number of output sources if odd then extract from the first source a sample more than the second
            # one
            if num_sources_out % 2 == 0:
                left_num_samples = same_num_samples
                right_num_samples = same_num_samples
            else:
                left_num_samples = same_num_samples + 1
                right_num_samples = same_num_samples

            if debug:
                print("Num left samples: {}".format(left_num_samples))
                print("Num right samples: {}".format(right_num_samples))

            # split data sources based on the sample ratios
            left_sources = self.data1.split_randomly(seed, num_split=left_num_samples, debug=debug)

            if right_num_samples == 1:
                right_sources = [self.data2]
            else:
                right_sources = self.data2.split_randomly(seed, num_split=right_num_samples, debug=debug)

            if debug:
                print("\nLeft samples")
                for sample_index, left_sample in enumerate(left_sources):
                    left_sample_data = left_sample.get_data()
                    print("SOURCE #{} (from source {})".format(sample_index, left_sample.get_source_id()))
                    print(left_sample_data.head())
                    print(left_sample_data.shape)
                print("\nRight samples")
                for sample_index, right_source in enumerate(right_sources):
                    right_source_data = right_source.get_data()
                    print("SOURCE #{} (from source {})".format(sample_index, right_source.get_source_id()))
                    print(right_source_data.head())
                    print(right_source_data.shape)

            # save split data sources
            for i in range(num_sources_out):
                if i % 2 == 0:
                    data_sources.append(left_sources[int(i / 2)])
                else:
                    data_sources.append(right_sources[int(i / 2)])

            if debug:
                print("\nALL")
                for source_index, source in enumerate(data_sources):
                    source_data = source.get_data()
                    print("SOURCE #{} (from source {})".format(source_index, source.get_source_id()))
                    print(source_data.head())
                    print(source_data.shape)

        else:  # mix

            # the two data sources are split in equi-sized samples and their records are mixed
            # left_sources = self.data1.split_randomly(seed=seed, sample_ratio=sample_ratio, debug=debug)
            left_sources = self.data1.split_randomly(seed=seed, num_split=num_sources_out, debug=debug)

            # right_sources = self.data2.split_randomly(seed=seed, sample_ratio=sample_ratio, debug=debug)
            right_sources = self.data2.split_randomly(seed=seed, num_split=num_sources_out, debug=debug)

            # combine samples from the first and the second data sources and generate output split data sources
            for i in range(len(left_sources)):
                left_source = left_sources[i]
                right_source = right_sources[i]

                if debug:
                    left_data = left_source.get_data()
                    print("SOURCE #{} (from source {})".format(i, left_source.get_source_id()))
                    print(left_data.head())
                    print(left_data.shape)
                    right_data = right_source.get_data()
                    print("SOURCE #{} (from source {})".format(i, right_source.get_source_id()))
                    print(right_data.head())
                    print(right_data.shape)

                # store in a tuple the two samples that will be combined in order to create a single output source
                out_source = (left_source, right_source)
                data_sources.append(out_source)

        return data_sources

    def split_randomly_data(self, num_sources_out, seed, mixed=False, debug=False):
        """
        This method splits randomly the original data sources and consequently also the integrated data. Two split
        techniques are available, which differ in mixing (or not) the information of the original data sources.

        :param num_sources_out: number of sources expected in output
        :param seed: seed for random sampling
        :param mixed: flag for selecting the split technique
        :param debug: flag for enabling debug modality
        :return: a list of data sources and a list of integrated data
        """

        # check input parameter data types
        if not isinstance(num_sources_out, int):
            raise TypeError("Wrong data type for parameter num_sources_out. Only integer data type is allowed.")

        if not isinstance(seed, int):
            raise TypeError("Wrong data type for parameter seed. Only integer data type is allowed.")

        if not isinstance(mixed, bool):
            raise TypeError("Wrong data type for parameter mixed. Only boolean data type is allowed.")

        # check input parameter data values
        if num_sources_out <= 2:
            raise ValueError("Wrong value for parameter num_sources_out. Provide a value grater than 2.")

        # STEP 1: split data sources
        data_sources = self.split_randomly_data_sources(num_sources_out, seed, mixed=mixed, debug=debug)

        # STEP 2: split integrated data based on data source split
        # extract from the integrated data the entities included in the first out source AND in the second out source
        if debug:
            print("\nINTEGRATED #0")

        initial_source1 = data_sources[0]
        initial_source2 = data_sources[1]

        if mixed:
            mix_intial_source1 = DataSource(pd.concat([initial_source1[0].get_data(), initial_source2[0].get_data()]),
                                            initial_source1[0].get_source_id())
            mix_intial_source2 = DataSource(pd.concat([initial_source1[1].get_data(), initial_source2[1].get_data()]),
                                            initial_source1[1].get_source_id())
            prev_integrated = self.integrated_data.filter_by_two_sources(mix_intial_source1, mix_intial_source2,
                                                                         debug=debug)

            out_source1 = DataSource(pd.concat([initial_source1[0].get_data(), initial_source1[1].get_data()]),
                                     self.data1.get_source_id())
            out_source2 = DataSource(pd.concat([initial_source2[0].get_data(), initial_source2[1].get_data()]),
                                     self.data2.get_source_id())
        else:
            prev_integrated = self.integrated_data.filter_by_two_sources(initial_source1, initial_source2, debug=debug)

            out_source1 = initial_source1
            out_source2 = initial_source2

        out_sources = [out_source1, out_source2]  # output list of sources
        out_integrated = [prev_integrated]  # output list of integrated data
        for i in range(2, len(data_sources)):

            if debug:
                print("\nINTEGRATED #{}".format(i - 1))

            current_data_source = data_sources[i]

            if mixed:  # mixed technique -> current_data_source is a tuple

                # combine samples from the first and the second data sources and generate a single output data source
                source1 = current_data_source[0]
                source2 = current_data_source[1]
                current_integrated = self.integrated_data.filter_by_two_sources(source1, source2, debug=debug)

                joint_source = DataSource(pd.concat([source1.get_data(), source2.get_data()]), i)
                out_sources.append(joint_source)

            else:  # non mixed technique

                joint_source = current_data_source

                # based on the provenance (first o second original data sources) extract from the integrated data only
                # the entities belonging to the currrent data source
                joint_source_id = joint_source.get_source_id()
                current_integrated = self.integrated_data.filter_by_source(joint_source_id,
                                                                           indexes=joint_source.get_ids(), debug=debug)

            out_sources.append(joint_source)

            # update the current integrated data by joining the information of the previous integrated data source
            cumulative_integrated_data = pd.concat([prev_integrated.get_data(), current_integrated.get_data()])
            cumulative_integrated = IntegratedDataset(cumulative_integrated_data, prev_integrated.id_col,
                                                      prev_integrated.source_id_col, prev_integrated.entity_label_col)

            out_integrated.append(cumulative_integrated)

            if debug:
                print("Shape: {}".format(cumulative_integrated.get_data().shape))

            prev_integrated = cumulative_integrated.copy()

        print("********** PREPARED DATA **********\n")
        for source_id, source in enumerate(out_sources):
            print("SOURCE{}: {}\n".format(source_id, source.get_data().shape))

        for integration_id, integration in enumerate(out_integrated):
            print("INTEGRATION{}: {}\n".format(integration_id, integration.get_data().shape))

        print("***********************************\n")

        return out_sources, out_integrated


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    # RESTAURANT EXAMPLE
    # example = 'restaurant'
    # EXAMPLE_DIR = os.path.join(DATA_DIR, example)
    # # source 1
    # file1 = os.path.join(EXAMPLE_DIR, "original", "zomato.csv")
    # source1_id = 0
    # # source 2
    # file2 = os.path.join(EXAMPLE_DIR, "original", "yelp.csv")
    # source2_id = 1
    # # integrated data
    # file3 = os.path.join(EXAMPLE_DIR, "original", "labeled_data.csv")

    # BIKE EXAMPLE
    # example = 'bike'
    # EXAMPLE_DIR = os.path.join(DATA_DIR, example)
    # # source 1
    # file1 = os.path.join(EXAMPLE_DIR, "original", "bikedekho.csv")
    # source1_id = 0
    # # source 2
    # file2 = os.path.join(EXAMPLE_DIR, "original", "bikewale.csv")
    # source2_id = 1
    # # integrated data
    # file3 = os.path.join(EXAMPLE_DIR, "original", "labeled_data.csv")

    # MOVIE EXAMPLE
    example = 'movie'
    EXAMPLE_DIR = os.path.join(DATA_DIR, example)
    # source 1
    file1 = os.path.join(EXAMPLE_DIR, "original", "rotten_tomatoes.csv")
    source1_id = 0
    # source 2
    file2 = os.path.join(EXAMPLE_DIR, "original", "imdb.csv")
    source2_id = 1
    # integrated data
    file3 = os.path.join(EXAMPLE_DIR, "original", "labeled_data.csv")

    # prepare data
    data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3)
    data_prep_comp.save_all_data(EXAMPLE_DIR)
    # original_data_sources, integrated_data = data_prep_comp.split_randomly_data(5, 24, mixed=True)
