import wget
import os
import pandas as pd
from uedi.data_integration.data_integration_utilities import clean_integrated_data, expand_integrated_dataset, get_sources
from uedi.utils.general_utilities import check_parameter_type
from uedi.utils.file_utilities import check_file_existence, create_dir
from uedi.data_integration.data_preparation import convert_matching_pairs_to_integrated_dataset
from uedi.data_integration.data_fusion import DataFusionComponent


class DeepMatcherRepoManager(object):
    """
    Deep Matcher data repository manager.
    """
    available_datasets = ['Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM', 'Structured_Amazon-Google',
                          'Structured_Walmart-Amazon', 'Structured_Beer', 'Structured_iTunes-Amazon',
                          'Structured_Fodors-Zagats', 'Textual_Abt-Buy', 'Textual_Company', 'Dirty_iTunes-Amazon',
                          'Dirty_DBLP-ACM', 'Dirty_DBLP-GoogleScholar', 'Dirty_Walmart-Amazon']
    repo_path = os.path.join(os.path.abspath(''), "data", "dm")

    @staticmethod
    def download_data(dataset_id: str):
        """
        This function downloads in the repository the dataset identified by the provided dataset id.

        :param dataset_id: dataset identifier
        :return: None
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')

        if dataset_id not in DeepMatcherRepoManager.available_datasets:
            raise ValueError("Wrong dataset name provided.")

        print("Downloading dataset...")

        dataset_id_tokens = dataset_id.split("_")
        data_type = dataset_id_tokens[0]
        dataset = '_'.join(dataset_id_tokens[1:])

        datasets_type_path = os.path.join(DeepMatcherRepoManager.repo_path, data_type)
        dataset_path = os.path.join(datasets_type_path, dataset)
        original_dataset_path = os.path.join(dataset_path, "original")

        create_dir(datasets_type_path)
        create_dir(dataset_path)
        create_dir(original_dataset_path)

        base_url = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/{}/{}/exp_data".format(data_type, dataset)

        # download table A
        url_table_a = "{}/tableA.csv".format(base_url)
        file_path_table_a = os.path.join(original_dataset_path, "tableA.csv")
        wget.download(url_table_a, file_path_table_a)

        # download table B
        url_table_b = "{}/tableB.csv".format(base_url)
        file_path_table_b = os.path.join(original_dataset_path, "tableB.csv")
        wget.download(url_table_b, file_path_table_b)

        # download train data and extend it with data sources info
        url_train = "{}/train.csv".format(base_url)
        file_path_train = os.path.join(original_dataset_path, "train.csv")
        wget.download(url_train, file_path_train)
        complete_train = expand_integrated_dataset(file_path_train, file_path_table_a, file_path_table_b)
        complete_train.to_csv(file_path_train, index=False)

        # download test data and extend it with data sources info
        url_test = "{}/test.csv".format(base_url)
        file_path_test = os.path.join(original_dataset_path, "test.csv")
        wget.download(url_test, file_path_test)
        complete_test = expand_integrated_dataset(file_path_test, file_path_table_a, file_path_table_b)
        complete_test.to_csv(file_path_test, index=False)

        # download validation data and extend it with data sources info
        url_valid = "{}/valid.csv".format(base_url)
        file_path_valid = os.path.join(original_dataset_path, "valid.csv")
        wget.download(url_valid, file_path_valid)
        complete_valid = expand_integrated_dataset(file_path_valid, file_path_table_a, file_path_table_b)
        complete_valid.to_csv(file_path_valid, index=False)

        print("Download completed successfully. Dataset inserted in the repository.")

    @staticmethod
    def check_dataset_existence(dataset_id: str):
        """
        This function checks if the dataset, identified by the provided dataset id, exists in the repository.
        If the dataset doesn't exist and a valid dataset id has been provided, then it is downloaded.

        :param dataset_id: dataset identifier
        :return: boolean indication of the existence of the dataset in the repository
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')

        if dataset_id not in DeepMatcherRepoManager.available_datasets:
            return False

        dataset_id_tokens = dataset_id.split("_")
        data_type = dataset_id_tokens[0]
        dataset = '_'.join(dataset_id_tokens[1:])

        missing_files = False
        dataset_dir = os.path.join(DeepMatcherRepoManager.repo_path, data_type, dataset, "original")
        if os.path.exists(dataset_dir):

            files = ['tableA.csv', 'tableB.csv', 'train.csv', 'test.csv', 'valid.csv']
            for f in files:
                if not os.path.exists(os.path.join(dataset_dir, f)):
                    missing_files = True
                    break
        else:
            missing_files = True

        if missing_files:
            DeepMatcherRepoManager.download_data(dataset_id)
        else:
            print("Dataset already stored in the repository.")

        return True

    @staticmethod
    def remove_duplicates(dataset_id: str, th: float, attributes: list = None, force: bool = False):
        """
        This function removes the duplicated records from train, test and validation files belonging to the provided
        dataset. If the cleaned version is already available and if the force flag is not set to true, the cleaning
        process is not applied.

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

        DeepMatcherRepoManager.check_dataset_existence(dataset_id)

        if th < 0 or th > 1:
            raise ValueError("Wrong threshold value.")

        dataset_id_tokens = dataset_id.split("_")
        data_type = dataset_id_tokens[0]
        dataset = '_'.join(dataset_id_tokens[1:])

        datasets_type_path = os.path.join(DeepMatcherRepoManager.repo_path, data_type)
        dataset_path = os.path.join(datasets_type_path, dataset)
        original_dataset_path = os.path.join(dataset_path, "original")
        clean_dataset_path = os.path.join(dataset_path, "clean")

        create_dir(clean_dataset_path)

        file_path_table_a = os.path.join(original_dataset_path, "tableA.csv")
        file_path_table_b = os.path.join(original_dataset_path, "tableB.csv")
        train_path = os.path.join(original_dataset_path, "train.csv")
        clean_train_path = os.path.join(clean_dataset_path, "train.csv")
        test_path = os.path.join(original_dataset_path, "test.csv")
        clean_test_path = os.path.join(clean_dataset_path, "test.csv")
        valid_path = os.path.join(original_dataset_path, "valid.csv")
        clean_valid_path = os.path.join(clean_dataset_path, "valid.csv")
        all_path_triplet = os.path.join(dataset_path, "{}_{}_triplet.csv".format(data_type, dataset))
        all_path = os.path.join(dataset_path, "{}_{}.csv".format(data_type, dataset))

        clean_files = [clean_train_path, clean_test_path, clean_valid_path, all_path_triplet, all_path]
        missing_clean_files = False
        for clean_file in clean_files:
            if not os.path.exists(clean_file):
                missing_clean_files = True
                break

        if not missing_clean_files:
            print("Dataset cleaned version already available.")

        if missing_clean_files or force:

            print("Cleaning dataset...")

            # clean train dataset
            clean_train = clean_integrated_data(train_path, th, attributes)
            clean_train.to_csv(clean_train_path, index=False)
            complete_train = expand_integrated_dataset(clean_train_path, file_path_table_a, file_path_table_b)
            complete_train.to_csv(clean_train_path, index=False)

            # clean test dataset
            clean_test = clean_integrated_data(test_path, th, attributes)
            clean_test.to_csv(clean_test_path, index=False)
            complete_test = expand_integrated_dataset(clean_test_path, file_path_table_a, file_path_table_b)
            complete_test.to_csv(clean_test_path, index=False)

            # clean validation dataset
            clean_valid = clean_integrated_data(valid_path, th, attributes)
            clean_valid.to_csv(clean_valid_path, index=False)
            complete_valid = expand_integrated_dataset(clean_valid_path, file_path_table_a, file_path_table_b)
            complete_valid.to_csv(clean_valid_path, index=False)

            # concat cleaned datasets and change data format
            # FIXME: check if this dataset has to be cleaned
            complete_dataset = pd.concat([complete_train, complete_test, complete_valid])
            complete_dataset.to_csv(all_path_triplet, index=False)

            all_s1, all_s2 = get_sources(complete_dataset)
            match_id = "id"
            left_id = "left_id"
            right_id = "right_id"
            match_label = "label"
            _, _, dataset_integrated_format = convert_matching_pairs_to_integrated_dataset(all_s1, 0, "id", all_s2, 1,
                                                                                           "id", complete_dataset,
                                                                                           match_id, left_id, right_id,
                                                                                           match_label)
            dataset_integrated_format.get_data().to_csv(all_path, index=False)

            print("Dataset cleaned successfully.")

        else:
            print("Clean dataset already stored in the repository.")

    @staticmethod
    def get_dataset_file(dataset_id: str, file_id: str, data_type: str):
        """
        This function retrieves a dataset file.

        :param dataset_id: dataset identifier
        :param file_id: file identifier. Available identifier: "train", "test", "valid", "all_triplet" or "all"
        :param data_type: file data type. Available data types: "original" or "clean"
        :return: Pandas DataFrame containing the requested file
        """

        check_parameter_type(dataset_id, 'dataset_id', str, 'str')
        check_parameter_type(file_id, 'file_id', str, 'str')
        check_parameter_type(data_type, 'data_type', str, 'str')

        if dataset_id not in DeepMatcherRepoManager.available_datasets:
            raise ValueError("Dataset identifier not found.")

        file_ids = ["train", "test", "valid", "all_triplet", "all", "tableA", "tableB"]
        if file_id not in file_ids:
            raise ValueError("File identifier not found. Available identifiers: {}.".format(file_ids))

        data_types = ["original", "clean"]
        if data_type not in data_types:
            raise ValueError("Wrong data type selected. Available data types: {}.".format(data_types))

        dataset_id_tokens = dataset_id.split("_")
        dataset_type = dataset_id_tokens[0]
        dataset = '_'.join(dataset_id_tokens[1:])

        if file_id == "all":
            if data_type == 'clean':
                data_type = ''
            file_id = "{}_{}".format(dataset_type, dataset)

        if file_id == "all_triplet":
            if data_type == 'clean':
                data_type = ''
            file_id = "{}_{}_triplet".format(dataset_type, dataset)

        if file_id == "tableA" or file_id == "tableB":
            if data_type == 'clean':
                raise ValueError("No {} clean version available.".format(file_id))

        file_path = os.path.join(DeepMatcherRepoManager.repo_path, dataset_type, dataset, data_type,
                                 "{}.csv".format(file_id))
        check_file_existence(file_path)

        return pd.read_csv(file_path)

    @staticmethod
    def combine_dataset_original_data(dataset1_id: str, dataset2_id: str, random_seed: int = 24):
        """
        This function combines two dataset files.

        :param dataset1_id: first dataset identifier
        :param dataset2_id: second dataset identifier
        :param random_seed: the seed for random choices
        :return: Pandas DataFrame containing the combination of the requested files
        """

        train1 = DeepMatcherRepoManager.get_dataset_file(dataset1_id, 'train', 'original')
        test1 = DeepMatcherRepoManager.get_dataset_file(dataset1_id, 'test', 'original')
        valid1 = DeepMatcherRepoManager.get_dataset_file(dataset1_id, 'valid', 'original')

        train2 = DeepMatcherRepoManager.get_dataset_file(dataset2_id, 'train', 'original')
        test2 = DeepMatcherRepoManager.get_dataset_file(dataset2_id, 'test', 'original')
        valid2 = DeepMatcherRepoManager.get_dataset_file(dataset2_id, 'valid', 'original')

        integration1 = pd.concat([train1, test1, valid1])
        integration2 = pd.concat([train2, test2, valid2])

        match_id = "id"
        left_id = "left_id"
        right_id = "right_id"
        match_label = "label"

        s11, s21 = get_sources(integration1)
        _, _, integration_container1 = convert_matching_pairs_to_integrated_dataset(s11, 0, "id", s21, 1,
                                                                                    "id", integration1,
                                                                                    match_id, left_id, right_id,
                                                                                    match_label)
        final_integration1 = integration_container1.get_data()
        max_entity_id1 = final_integration1['entity_id'].max()


        s12, s22 = get_sources(integration2)
        _, _, integration_container2 = convert_matching_pairs_to_integrated_dataset(s12, 0, "id", s22, 1,
                                                                                    "id", integration2,
                                                                                    match_id, left_id, right_id,
                                                                                    match_label)
        final_integration2 = integration_container2.get_data()
        final_integration2['entity_id'] = final_integration2['entity_id'] + max_entity_id1

        s1 = pd.concat([s11, s12])
        s2 = pd.concat([s21, s22])
        integration = pd.concat([final_integration1, final_integration2])

        integration_container2.set_data(integration)
        data_fusion_comp = DataFusionComponent(integration_container2)
        compact_integration = data_fusion_comp.select_random_attribute_values(random_seed)

        return s1, s2, compact_integration.get_data()

