from uedi.data_repos.dm_utilities import DeepMatcherRepoManager
from uedi.data_repos.cora_utilities import CoraRepoManager
from uedi.data_integration.data_integration_utilities import dataset_tokenizer
from uedi.utils.general_utilities import check_parameter_type


class DataManager(object):
    """
    This class implements the manager of the data pre-processing operations.
    """

    def __init__(self, repo_id: str, dataset_id: str):
        """
        This method manages the initialization of the data pre-processing manager variables.

        :param repo_id: repository identifier
        :param dataset_id: dataset identifier
        """

        check_parameter_type(repo_id, 'repo_id', str, 'string')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        available_repos = ["deep-matcher", "cora"]
        if repo_id not in available_repos:
            raise ValueError("Wrong repo name provided. Only these values are allowed: {}".format(available_repos))

        repo = None
        if repo_id == 'deep-matcher':
            repo = DeepMatcherRepoManager
        elif repo_id == "cora":
            repo = CoraRepoManager

        if not repo.check_dataset_existence(dataset_id):
            raise ValueError("Wrong dataset name provided.")

        self.repo = repo
        self.dataset_id = dataset_id

    def get_dataset_file(self, file_id: str, data_type: str):
        """
        This method demands to the selected repository manager to retrieve a dataset file.

        :param file_id: file identifier. Available identifier: "train", "test", "valid" or "all"
        :param data_type: file data type. Available data types: "original" or "clean"
        :return: Pandas DataFrame containing the requested file
        """

        check_parameter_type(file_id, 'file_id', str, 'string')
        check_parameter_type(data_type, 'data_type', str, 'string')

        return self.repo.get_dataset_file(dataset_id=self.dataset_id, file_id=file_id, data_type=data_type)

    def remove_duplicates(self, th: float, attributes: list = None, force: bool = False):
        """
        This method demands to the selected repository manager to create a cleaned version of the specified dataset.

        :param th: threshold value for the deduplication process
        :param attributes: optional attributes where to limit the deduplication process
        :param force: boolean flag that indicates to force the execution of the cleaning task also if the cleaned
                      version is already available
        :return: None
        """

        check_parameter_type(th, 'th', float, 'float')
        check_parameter_type(attributes, 'attributes', list, 'list', optional_param=True)
        check_parameter_type(force, 'force', bool, 'boolean')

        if th < 0 or th > 1:
            raise ValueError("Wrong threshold value.")

        self.repo.remove_duplicates(dataset_id=self.dataset_id, th=th, attributes=attributes)

    def numerical_binning(self):
        pass

    def tokenization(self, file_id: str, data_type: str, attributes: list = None):
        """
        This method tokenizes the specified dataset file.

        :param file_id: file identifier. Available identifier: "train", "test", "valid" or "all"
        :param data_type: file data type. Available data types: "original" or "clean"
        :param attributes: optional attributes where to limit the tokenization process
        :return: Pandas DataFrame containing the tokenized file
        """

        dataset = self.get_dataset_file(file_id=file_id, data_type=data_type)

        return dataset_tokenizer(dataset, attributes)