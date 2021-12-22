import pandas as pd
import os

from uedi.data_repos.data_manager import DataManager
from uedi.utils.general_utilities import check_parameter_type, check_cols_in_dataframe
from uedi.utils.file_utilities import create_dir


class ERGroundTruthValidator(object):
    """
    This class checks whether a EM ground truth is applicable also for ER tasks, where entities are represented as
    cliques (i.e., fully connected structures of matching records).
    """

    def __init__(self):
        """
        This method initializes the state of the validator.
        """
        self.validity_info_history = []
        self.gt_validity_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'gt_validity')

    def check_validity(self, data: pd.DataFrame, dataset_id: str):
        """
        This method checks the ground truth validity in terms of completeness (i.e., cliques are complete) and
        correctness (i.e., no conflicting information).

        :param data: Pandas DataFrame containing the ground truth to validate
        :param dataset_id: identifier of the dataset
        return: dictionary containing some statistics about invalid ground truth elements
        """

        check_parameter_type(data, 'data', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        col_left = 'ltable_id'
        col_right = 'rtable_id'
        col_label = 'label'

        check_cols_in_dataframe(data, [col_left, col_right, col_label])

        list_idx = data.loc[data[col_label] == 1, col_left].unique()

        total_res = {'dataset_id': dataset_id, 'completeness': set([]), 'correctness': set([]), 'num_entities': 0,
                     'num_incomplete_entities': 0, 'num_incorrect_entities': 0, 'total_pairs': len(data)}
        complete_entities = {}

        # loop over left records
        for idx in list_idx:
            # Define candidate left and right set
            candidate_left = set()
            candidate_right = set()

            # discover all the records included in clique that contains the current left record
            explore_left = set([idx])
            while explore_left:

                # get the right records that directly match with the current left record
                cond = (data[col_left].isin(explore_left)) & (data[col_label] == 1)
                explore_right = set(data.loc[cond, col_right].to_list())

                # consider unseen right ids only
                explore_right = explore_right.difference(candidate_right)

                # save initial left record in the final results
                candidate_left = candidate_left.union(explore_left)

                # get the left records that directly match with the previously identified right records
                cond = (data[col_right].isin(explore_right)) & (data[col_label] == 1)
                explore_left = set(data.loc[cond, col_left].to_list())

                # consider unseen left ids only
                explore_left = explore_left.difference(candidate_left)

                # save right records in the final results
                candidate_right = candidate_right.union(explore_right)

                # continue (if any) with the next level matching records

            # loop over all pairs of records included in the clique and discover incomplete/incorrect matching relations
            found_incomplete = False
            found_incorrect = False
            for i in candidate_left:
                for j in candidate_right:

                    # check completeness: missing a matching relations in the clique
                    cond = (data[col_left] == i) & (data[col_right] == j) & (data[col_label] == 1)
                    if data[cond].empty:
                        total_res['completeness'].add((i, j))
                        found_incomplete = True

                    # check correctness: discover non-match relations in the clique
                    cond = (data[col_left] == i) & (data[col_right] == j) & (data[col_label] == 0)
                    if len(data[cond]) > 0:
                        total_res['correctness'].add((i, j))
                        found_incorrect = True

            if (tuple(candidate_left), tuple(candidate_right)) not in complete_entities:
                if found_incomplete:
                    total_res['num_incomplete_entities'] += 1
                if found_incorrect:
                    total_res['num_incorrect_entities'] += 1

            complete_entities[(tuple(candidate_left), tuple(candidate_right))] = 1

        # add some statistics to the results
        total_res['num_entities'] = len(complete_entities)
        total_res['completeness'] = total_res['completeness'].difference(total_res['correctness'])
        total_res['num_incomplete_pairs'] = len(total_res['completeness'])
        total_res['num_incorrect_pairs'] = len(total_res['correctness'])
        total_res['num_incomplete_entities'] -= total_res['num_incorrect_entities']
        incomplete_pairs_ratio = "{:.2f}".format((len(total_res['completeness']) / total_res['total_pairs']) * 100)
        total_res['incomplete_pairs_ratio'] = incomplete_pairs_ratio
        incomplete_entities_ratio = "{:.2f}".format(
            (total_res['num_incomplete_entities'] / total_res['num_entities']) * 100)
        total_res['incomplete_entities_ratio'] = incomplete_entities_ratio
        incorrect_pairs_ratio = "{:.2f}".format((len(total_res['correctness']) / total_res['num_entities']) * 100)
        total_res['incorrect_pairs_ratio'] = incorrect_pairs_ratio
        incorrect_entities_ratio = "{:.2f}".format(
            (total_res['num_incorrect_entities'] / total_res['num_entities']) * 100)
        total_res['incorrect_entities_ratio'] = incorrect_entities_ratio
        total_res["num_wrong_entities"] = total_res["num_incomplete_entities"] + total_res["num_incorrect_entities"]
        total_res["wrong_entities_ratio"] = "{:.2f}".format(
            (total_res["num_wrong_entities"] / total_res['num_entities']) * 100)

        results = pd.DataFrame([total_res])

        short_results = results.copy()
        del short_results['completeness']
        del short_results['correctness']

        self.validity_info_history.append(short_results)

        return results

    def save_results(self):
        """
        This method saves all the ground truth checks included in the history.
        """

        create_dir(self.gt_validity_dir)
        res = pd.concat(self.validity_info_history)
        res.to_csv(os.path.join(self.gt_validity_dir, 'validity_results.csv'), index=False)


if __name__ == '__main__':

    repository_id = 'deep-matcher'

    dataset_ids = ['Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM', 'Structured_Amazon-Google',
                   'Structured_Walmart-Amazon', 'Structured_Beer', 'Structured_iTunes-Amazon',
                   'Structured_Fodors-Zagats', 'Textual_Abt-Buy', 'Dirty_iTunes-Amazon', 'Dirty_DBLP-ACM',
                   'Dirty_DBLP-GoogleScholar', 'Dirty_Walmart-Amazon']

    gt_validator = ERGroundTruthValidator()

    for dataset_id in dataset_ids:
        print(dataset_id)
        pre_data_manager = DataManager(repository_id, dataset_id)

        train = pre_data_manager.get_dataset_file("train", "original")
        valid = pre_data_manager.get_dataset_file("valid", "original")
        test = pre_data_manager.get_dataset_file("test", "original")

        data = pd.concat([train, valid, test])

        gt_validator.check_validity(data, dataset_id)

    gt_validator.save_results()
