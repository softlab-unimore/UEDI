import pandas as pd
from collections import Counter
import os

from uedi.utils.general_utilities import check_parameter_type
from uedi.utils.file_utilities import check_file_existence


class CoraRepoManager(object):
    """
    Cora data repository manager.
    """
    available_datasets = ['cora']
    repo_path = os.path.join(os.path.abspath(''), "data", "cora")

    @staticmethod
    def check_dataset_existence(dataset_id: str):
        """
        This function checks if the dataset, identified by the provided dataset id, exists in the repository.

        :param dataset_id: dataset identifier
        :return: boolean indication of the existence of the dataset in the repository
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')

        if dataset_id not in CoraRepoManager.available_datasets:
            return False

        return True

    @staticmethod
    def remove_duplicates(dataset_id: str, th: float, attributes: list = None, force: bool = False):
        """
        This function clean the dataset with the provided dataset_id.

        :param dataset_id: dataset identifier
        :param th: threshold value for the deduplication process
        :param attributes: optional list of attributes to be considered in the deduplication process
        :param force: boolean flag that indicates to force the execution of the cleaning task also if the cleaned
                      version is already available
        :return: None
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')
        check_parameter_type(th, 'th', float, 'float')
        check_parameter_type(attributes, 'attributes', list, 'list', optional_param=True)
        check_parameter_type(force, 'force', bool, 'boolean')

        if not CoraRepoManager.check_dataset_existence(dataset_id):
            raise ValueError("Dataset identifier found.")

        if th < 0 or th > 1:
            raise ValueError("Wrong threshold value.")

        print("Clean dataset already stored in the repository.")

    @staticmethod
    def get_dataset_file(dataset_id: str, file_id: str, data_type: str):
        """
        This function retrieves a dataset file.

        :param dataset_id: dataset identifier
        :param file_id: file identifier. Available identifier: "all"
        :param data_type: file data type. Available data types: "original"
        :return: Pandas DataFrame containing the requested file
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')
        check_parameter_type(file_id, 'file_id', str, 'str')
        check_parameter_type(data_type, 'data_type', str, 'str')

        if dataset_id not in CoraRepoManager.available_datasets:
            raise ValueError("Dataset identifier not found.")

        file_ids = ["all"]
        if file_id not in file_ids:
            raise ValueError("File identifier not found. Available identifiers: {}.".format(file_ids))

        data_types = ["original"]
        if data_type not in data_types:
            raise ValueError("Wrong data type selected. Available data types: {}.".format(data_types))

        file_path = os.path.join(CoraRepoManager.repo_path, "cora.csv")
        check_file_existence(file_path)

        return pd.read_csv(file_path, sep="\t")


def get_multi_data_sources(data: pd.DataFrame):
    """
    This function extract multiple data sources from the input integrated dataset.

    :param data: Pandas DataFrame containing the dataset where the data sources will be extracted
    :return: list of Pandas DataFrames containing multiple data sources
    """

    # count the number of records that refer to the same entity
    counter = Counter(data['entity_id'])
    data['count'] = data['entity_id'].map(counter)

    # get the number of records that refer to the largest entity
    count_max_entity = data['count'].max()

    # extract multiple data sources from the ground truth
    sources = [[] for _ in range(count_max_entity)]
    for i in range(1, count_max_entity + 1):

        # get the entities that are referred by a number of records equal to 'i'
        entities_grouped_by_count = data[data['count'] == i]
        if entities_grouped_by_count.empty:
            continue

        # loop over the unique entities that satisfy the previous condition and split their records into multiple
        # data sources
        for id in entities_grouped_by_count['entity_id'].unique():
            records = entities_grouped_by_count[entities_grouped_by_count['entity_id'] == id]
            for j in range(len(records)):
                sources[j].append(records.iloc[j])

    sources = [pd.DataFrame(x) for x in sources]

    return sources
