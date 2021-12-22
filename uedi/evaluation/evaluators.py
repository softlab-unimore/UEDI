import os
import time
import pandas as pd
import numpy as np

from uedi.utils.file_utilities import create_dir
from uedi.utils.general_utilities import check_parameter_type, check_cols_in_dataframe, get_regression_metric_scores
from uedi.data_integration.data_integration_utilities import get_three_sources, dataset_tokenizer, get_integrated_s1_s2, \
    create_integration_dataset_by_data_type
from uedi.metrics.metrics import TwoWayMetricCalculator
from uedi.nlp.language_models import LanguageModel
from uedi.evaluation.scenarios import DataIntegrationScenarioContainer
from uedi.models.data_models import IntegratedDataset
from uedi.data_integration.data_fusion import DataFusionComponent


class MultiEntityTypesEvaluator(object):
    """
    This class evaluates a metric over a set of data integration scenarios where the entity type is varied.
    """

    def __init__(self, dataset: pd.DataFrame, th: float = 0.4, columns: list = None):
        """
        This method initializes the evaluator by generating, from the input dataset, multiple data integration scenarios
        with variable entity types.

        :param dataset: Pandas DataFrame containing the dataset from which the scenarios will be generated
        :param th: threshold value for cleaning the data in the scenarios
        :param columns: optional list of columns to consider in the data
        """

        print("\nMULTI-ENTITY-TYPES EVALUATOR")

        check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(th, 'th', float, 'float')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)

        if th < 0 or th > 1:
            raise ValueError("Wrong value for parameter th. Only values in the range [0, 1] are allowed.")

        if columns is not None:
            check_cols_in_dataframe(dataset, columns)

        print("\nCreating scenarios...")
        # create 4 data sources from the input dataset
        s1, s2, s3 = get_three_sources(dataset)

        for eid in s2['entity_id']:
            assert eid in s1['entity_id'].values

        for eid in s3['entity_id']:
            assert eid not in s1['entity_id'].values

        s4 = pd.concat([s2, s3], ignore_index=True)
        print("|S4| = {}".format(len(s4)))

        # FIXME: is this step needed? why only s3 is cleaned? get_duplicate or clean?
        # clean and tokenize the data sources
        # only for check: get_duplicate(s3, th, columns=columns)
        # s1 = clean_single_data(s1, th, columns=columns)
        # s2 = clean_single_data(s2, th, columns=columns)
        # s3 = clean_single_data(s3, th, columns=columns)

        s1_records = dataset_tokenizer(s1, columns=columns)
        s2_records = dataset_tokenizer(s2, columns=columns)
        s3_records = dataset_tokenizer(s3, columns=columns)
        s4_records = dataset_tokenizer(s4, columns=columns)

        # save some common scenario parameters
        common_scenario_params = {}
        common_scenario_params["D1"] = len(s1)
        common_scenario_params["D2"] = len(s2)
        common_scenario_params["D3"] = len(s3)
        common_scenario_params["D4"] = len(s2) + len(s3)

        lm_s1 = LanguageModel(n=1, mtype='mle')
        lm_s1.fit(s1_records)
        common_scenario_params["V1"] = len(lm_s1.get_vocabs())

        lm_s2 = LanguageModel(n=1, mtype='mle')
        lm_s2.fit(s2_records)
        common_scenario_params["V2"] = len(lm_s2.get_vocabs())

        lm_s3 = LanguageModel(n=1, mtype='mle')
        lm_s3.fit(s3_records)
        common_scenario_params["V3"] = len(lm_s3.get_vocabs())

        lm_s4 = LanguageModel(n=1, mtype='mle')
        lm_s4.fit(s4_records)
        common_scenario_params["V4"] = len(lm_s4.get_vocabs())

        # create scenario 1
        s1_s2_concat = pd.concat([s1, s2], ignore_index=True)
        # s1_s2_match = get_integrated_s1_s2(s1_s2_concat, option=0)
        # s1_s2_match_records = dataset_tokenizer(s1_s2_match, columns=self.columns)

        s1_s2_match_records = s1_records
        s1_s2_concat_records = dataset_tokenizer(s1_s2_concat, columns=columns)
        scenario1_sources = [s1_records, s2_records]

        scenario1_data = [
            {
                'sources': scenario1_sources,
                'integrations': [s1_records, s1_s2_match_records],
                'data_type': 'match',
                'sources_ids': ['1', '2'],
                'integrations_ids': ['1', '1']
            },
            {
                'sources': scenario1_sources,
                'integrations': [s1_records, s1_s2_concat_records],
                'data_type': 'concat',
                'sources_ids': ['1', '2'],
                'integrations_ids': ['1', '1-2-concat']
            }
        ]
        scenario1_params = {'scenario': 1}
        scenario1_params.update(common_scenario_params)
        scenario1 = DataIntegrationScenarioContainer(scenario1_params, scenario1_data)

        # create scenario 2
        s1_s3_concat = pd.concat([s1, s3], ignore_index=True)
        s1_s3_match_records = s1_records
        s1_s3_concat_records = dataset_tokenizer(s1_s3_concat, columns=columns)
        scenario2_sources = [s1_records, s3_records]

        scenario2_data = [
            {
                'sources': scenario2_sources,
                'integrations': [s1_records, s1_s3_match_records],
                'data_type': 'match',
                'sources_ids': ['1', '3'],
                'integrations_ids': ['1', '1']
            },
            {
                'sources': scenario2_sources,
                'integrations': [s1_records, s1_s3_concat_records],
                'data_type': 'concat',
                'sources_ids': ['1', '3'],
                'integrations_ids': ['1', '1-3-concat']
            }
        ]
        scenario2_params = {'scenario': 2}
        scenario2_params.update(common_scenario_params)
        scenario2 = DataIntegrationScenarioContainer(scenario2_params, scenario2_data)

        # create scenario 3
        s1_s2_match = get_integrated_s1_s2(s1_s2_concat, option=0)
        s1_s4_match = s1_s2_match.copy()
        s1_s4_concat = pd.concat([s1, s4], ignore_index=True)
        s1_s4_perfect = s1_s3_concat.copy()
        s1_s4_match_records = dataset_tokenizer(s1_s4_match, columns=columns)
        s1_s4_concat_records = dataset_tokenizer(s1_s4_concat, columns=columns)
        s1_s4_perfect_records = dataset_tokenizer(s1_s4_perfect, columns=columns)
        scenario3_sources = [s1_records, s4_records]

        scenario3_data = [
            {
                'sources': scenario3_sources,
                'integrations': [s1_records, s1_s4_perfect_records],
                'data_type': 'perfect',
                'sources_ids': ['1', '4'],
                'integrations_ids': ['1', '1-4-perfect']
            },
            {
                'sources': scenario3_sources,
                'integrations': [s1_records, s1_s4_match_records],
                'data_type': 'match',
                'sources_ids': ['1', '4'],
                'integrations_ids': ['1', '1-4-match']
            },
            {
                'sources': scenario3_sources,
                'integrations': [s1_records, s1_s4_concat_records],
                'data_type': 'concat',
                'sources_ids': ['1', '4'],
                'integrations_ids': ['1', '1-4-concat']
            }
        ]
        scenario3_params = {'scenario': 3}
        scenario3_params.update(common_scenario_params)
        scenario3 = DataIntegrationScenarioContainer(scenario3_params, scenario3_data)

        print("\nScenarios created successfully.")

        self.scenarios = [scenario1, scenario2, scenario3]
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-entity-types')

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """
        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            # scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("\n{} evaluation ({})".format(data_type, metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


# class MultiMatchPercentagesEvaluator(object):
#     """
#     This class evaluates a metric over a set of data integration scenarios where the percentage of correctly predicted
#     matching/non-matching records is varied.
#     """
#     # repo_path = os.path.join(os.path.abspath('.'), "data", "scenarios", "multi-match-percentages")
#
#     def __init__(self, dataset: pd.DataFrame, match_non_match_ratios: list = None, num_scenarios: int = None,
#                  columns: list = None):
#         """
#         This method initializes the evaluator by generating, from the input dataset, multiple data integration scenarios
#         with variable correctly predicted matching/non-matching percentages. These percentages can be provided by the
#         user (i.e., match_non_match_ratios parameter) or automatically generated (in this case the user has to indicate
#         its number via the num_scenarios parameter).
#
#         :param dataset: Pandas DataFrame containing the dataset from which the scenarios will be generated
#         :param match_non_match_ratios: optional list of correctly predicted match-non_match percentages to consider
#         :param num_scenarios: number of combinations of correctly predicted match-non_match percentages to automatically
#                               generate
#         :param columns: optional list of columns to consider in the data
#         """
#
#         print("\nMULTI-MATCH-PERCENTAGES EVALUATOR")
#
#         check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
#         check_parameter_type(match_non_match_ratios, 'match_non_match_ratios', list, 'list', optional_param=True)
#         check_parameter_type(num_scenarios, 'num_scenarios', int, 'integer', optional_param=True)
#         check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
#
#         if match_non_match_ratios is None and num_scenarios is None:
#             raise ValueError("Provide one of the following parameters: 'match_non_match_ratios', 'num_scenarios'.")
#
#         if match_non_match_ratios is not None:
#             for match_non_match_ratio in match_non_match_ratios:
#                 check_parameter_type(match_non_match_ratio, 'match_non_match_ratios items', tuple, 'tuple')
#
#         if num_scenarios is not None:
#             if num_scenarios > 100:
#                 raise ValueError("Wrong value for parameter num_scenarios.")
#
#             # automatically create the match/non_match percentages by randomly selecting values in the range [0, 1] with
#             # step 0.1
#             match_non_match_ratios_set = set([])
#             match_non_match_ratios_set.add((1.0,1.0))
#             values_for_sampling = np.linspace(0,1,11)
#             np.random.seed(24)
#
#             while len(match_non_match_ratios_set) < num_scenarios:
#                 match_ratio = np.random.choice(values_for_sampling)
#                 non_match_ratio = np.random.choice(values_for_sampling)
#                 match_non_match = (float("{:.2f}".format(match_ratio)), float("{:.2f}".format(non_match_ratio)))
#
#                 match_non_match_ratios_set.add(match_non_match)
#
#             match_non_match_ratios = list(match_non_match_ratios_set)
#
#         print("#MATCH-NON_MATCH SCENARIOS: {}".format(len(match_non_match_ratios)))
#         print("MATCH-NON_MATCH SCENARIOS: {}".format(match_non_match_ratios))
#
#         true_all_data1, true_all_data2, true_all_integrated_dataset = create_compact_integration_dataset_by_data_type(
#             dataset, 'all', 'label')
#         true_data = true_all_integrated_dataset.data
#         true_matching_entities = len(true_data[true_data["match"] == 1])
#         true_non_matching_entities = len(true_data[true_data["match"] == 0])
#
#         self.scenarios = []
#
#         print("Creating scenarios...")
#         # create scenarios
#         # create a scenario for each user-provided correctly predicted match/non-match percentage pair
#         for scenario_id, match_non_match_ratio in enumerate(match_non_match_ratios, 1):
#
#             match_ratio = match_non_match_ratio[0]
#             non_match_ratio = match_non_match_ratio[1]
#             print("Scenario {} (match={}, non_match={})".format(scenario_id, match_ratio, non_match_ratio))
#
#             # create a matching prediction that satisfies the user-provided correctly predicted match and non-match
#             # ratios
#             predictions = generate_matching_predictions(dataset, "label", match_ratio=match_ratio,
#                                                         non_match_ratio=non_match_ratio)
#
#             # create a compact integrated dataset by considering all the data
#             all_data1, all_data2, all_integrated_dataset = create_compact_integration_dataset_by_data_type(predictions,
#                                                                                                            'all',
#                                                                                                            'pred')
#
#             tokenized_data1_all = dataset_tokenizer(all_data1.get_data(), columns=columns)
#             tokenized_data2_all = dataset_tokenizer(all_data2.get_data(), columns=columns)
#             tokenized_integrated_all = dataset_tokenizer(all_integrated_dataset.get_data(), columns=columns)
#
#             # collect the scenario parameters
#             scenario_params = {"scenario": scenario_id, "match_ratio": match_ratio, "non_match_ratio": non_match_ratio}
#
#             # get some other scenario parameters (e.g., evaluate the real prediction effectiveness)
#             other_params = get_binary_classification_effectiveness_report(list(predictions["label"].values),
#                                                                           list(predictions["pred"].values),
#                                                                           flat=True)
#             scenario_params.update(other_params)
#
#             ds_a, ds_b = get_sources(predictions)
#             cm_s1, cm_s2 = get_concat_and_compression_scores(ds_a, ds_b, predictions)
#             s1_params = get_effectiveness_metrics_from_confusion_matrix(cm_s1, {'1': 'match', '0': 'non_match'})
#             s2_params = get_effectiveness_metrics_from_confusion_matrix(cm_s2, {'1': 'match', '0': 'non_match'})
#             new_s1_params = {'s1_{}'.format(k): v for k, v in s1_params.items()}
#             new_s2_params = {'s2_{}'.format(k): v for k, v in s2_params.items()}
#             scenario_params.update(new_s1_params)
#             scenario_params.update(new_s2_params)
#
#             match_non_match_sizes = get_match_non_match_sizes(dataset, "label")
#             scenario_params.update(match_non_match_sizes)
#
#             pred_data = all_integrated_dataset.data
#             pred_matching_entities = len(pred_data[pred_data["match"] == 1])
#             pred_non_matching_entities = len(pred_data[pred_data["match"] == 0])
#
#             match_entities_variation = true_matching_entities - pred_matching_entities
#             non_match_entities_variation = true_non_matching_entities - pred_non_matching_entities
#             match_non_match_variation_data = {"delta_match_entities": match_entities_variation,
#                                               "delta_non_match_entities": non_match_entities_variation}
#             scenario_params.update(match_non_match_variation_data)
#
#             # # create a compact integrated dataset by considering the matching data only
#             # data1_match, data2_match, integrated_dataset_match = create_compact_integration_dataset_by_data_type(
#             #     predictions, 'match', 'pred')
#             # tokenized_data1_match = dataset_tokenizer(data1_match.get_data(), columns=columns)
#             # tokenized_data2_match = dataset_tokenizer(data2_match.get_data(), columns=columns)
#             # tokenized_integrated_match = dataset_tokenizer(integrated_dataset_match.get_data(), columns=columns)
#
#             # match_sources = [tokenized_data1_match, tokenized_data2_match]
#             # match_integrations = [tokenized_integrated_match, tokenized_integrated_match]
#             # match_sources = [tokenized_data1_all, tokenized_data2_all]
#             # match_integrations = [tokenized_integrated_all, tokenized_integrated_all]
#
#             # # create a compact integrated dataset by considering the non-matching data only
#             # data1_concat, data2_concat, integrated_dataset_concat = create_compact_integration_dataset_by_data_type(
#             #     predictions, 'non_match', 'pred')
#             # tokenized_data1_concat = dataset_tokenizer(data1_concat.get_data(), columns=columns)
#             # tokenized_data2_concat = dataset_tokenizer(data2_concat.get_data(), columns=columns)
#             # tokenized_integrated_concat = dataset_tokenizer(integrated_dataset_concat.get_data(), columns=columns)
#
#             # concat_sources = [tokenized_data1_concat, tokenized_data2_concat]
#             # concat_integrations = [tokenized_integrated_concat, tokenized_integrated_concat]
#             # concat_sources = [tokenized_data1_all, tokenized_data2_all]
#             # concat_integrations = [tokenized_integrated_all, tokenized_integrated_all]
#
#             all_sources = [tokenized_data1_all, tokenized_data2_all]
#             all_integrations = [tokenized_integrated_all, tokenized_integrated_all]
#
#             # create the scenario object
#             scenario_data = [
#                 # {
#                 #     'sources': match_sources,
#                 #     'integrations': match_integrations,
#                 #     'data_type': 'match',
#                 #     'sources_ids': ['data1_match_{}_{}'.format(match_ratio, non_match_ratio),
#                 #                     'data2_match_{}_{}'.format(match_ratio, non_match_ratio)],
#                 #     'integrations_ids': ['integration_match_{}_{}'.format(match_ratio, non_match_ratio),
#                 #                          'integration_match_{}_{}'.format(match_ratio, non_match_ratio)]
#                 # },
#                 # {
#                 #     'sources': concat_sources,
#                 #     'integrations': concat_integrations,
#                 #     'data_type': 'concat',
#                 #     'sources_ids': ['data1_non_match_{}_{}'.format(match_ratio, non_match_ratio),
#                 #                     'data2_non_match_{}_{}'.format(match_ratio, non_match_ratio)],
#                 #     'integrations_ids': ['integration_non_match_{}_{}'.format(match_ratio, non_match_ratio),
#                 #                          'integration_non_match_{}_{}'.format(match_ratio, non_match_ratio)]
#                 # },
#                 {
#                     'sources': all_sources,
#                     'integrations': all_integrations,
#                     'data_type': 'all',
#                     'sources_ids': ['data1_all_{}_{}'.format(match_ratio, non_match_ratio),
#                                     'data2_all_{}_{}.format(match_ratio, non_match_ratio)'.format(match_ratio,
#                                                                                                   non_match_ratio)],
#                     'integrations_ids': ['integration_all_{}_{}'.format(match_ratio, non_match_ratio),
#                                          'integration_all_{}_{}'.format(match_ratio, non_match_ratio)]
#                 }
#             ]
#             scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
#
#             self.scenarios.append(scenario)
#
#         self.results = None
#         self.results_dir = os.path.join(os.path.abspath('.'), 'data', 'output', 'results', 'multi-match-percentages')
#
#         print("Scenarios created successfully.")
#
#     def save_results(self, file_prefix: str):
#         """
#         This method saves the results to disk.
#
#         :param file_prefix: the prefix to be added to result file names
#         """
#
#         check_parameter_type(file_prefix, 'file_prefix', str, 'string')
#         create_dir(self.results_dir)
#
#         file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
#         self.results.to_csv(file_name, index=False)
#
#     def get_scenarios(self):
#         """
#         This method returns the generated scenarios.
#
#         :return: list of scenario objects
#         """
#
#         return self.scenarios
#
#     def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
#         """
#         This method evaluates the user-provided metric over the generated scenarios.
#
#         :param metric_calculator: the metric to evaluate
#         :param metric_params: a list of metric configurations
#         :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
#         :return: Pandas DataFrame containing the results of the evaluation
#         """
#
#         # check input parameters
#         check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
#         check_parameter_type(metric_params, 'metric_params', list, 'list')
#         check_parameter_type(mean, 'mean', bool, 'boolean')
#
#         # loop over the scenarios
#         for i, scenario in enumerate(self.scenarios, 1):
#             print()
#             print('---- Scenario {} ----'.format(i))
#             print()
#
#             scenario_params = scenario.get_params()
#             scenario_data = scenario.get_data()
#
#             # loop over the multiple metric configurations
#             for metric_param in metric_params:
#
#                 # evaluate the metric over multiple data types
#                 scenario_scores = []
#                 for scenario_item in scenario_data:
#                     sources = scenario_item['sources']
#                     integrations = scenario_item['integrations']
#                     data_type = scenario_item['data_type']
#                     sources_ids = scenario_item['sources_ids']
#                     integrations_ids = scenario_item['integrations_ids']
#
#                     print("{} evaluation (match={}, non_match={}, {})".format(data_type, scenario_params['match_ratio'],
#                                                                               scenario_params['non_match_ratio'],
#                                                                               metric_param))
#
#                     # calculate the metric score
#                     scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
#                                                                                       sources_ids, integrations_ids,
#                                                                                       metric_param, mean)
#
#                     scenario_scores.append(scenario_score)
#
#                 # save scenario scores
#                 export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
#                 scenario.save_metric_scores(export_metric_param, scenario_scores, mean)
#
#         results = []
#         for scenario in self.scenarios:
#             all_scores = scenario.get_scores()
#             for score_key in all_scores:
#                 results += all_scores[score_key]
#
#         self.results = pd.DataFrame(results)
#
#         return self.results

class MultiMatchPercentagesEvaluator(object):
    """
    This class evaluates a metric over a set of data integration scenarios where the percentage of correctly predicted
    matching/non-matching records is varied.
    """

    def __init__(self, dataset: pd.DataFrame, match_non_match_ratios: list, columns: list = None,
                 random_state: int = 24):
        """
        This method initializes the evaluator by generating, from the input dataset, multiple data integration scenarios
        with variable correctly predicted matching/non-matching percentages.

        :param dataset: Pandas DataFrame containing the dataset from which the scenarios will be generated
        :param match_non_match_ratios: list of correctly predicted match-non_match percentages to consider
        :param columns: optional list of columns to consider in the data
        :param random_state: the seed for the random choices
        """

        print("\nMULTI-MATCH-PERCENTAGES EVALUATOR")

        # check input parameter data types
        check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(match_non_match_ratios, 'match_non_match_ratios', list, 'list')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        for match_non_match_ratio in match_non_match_ratios:
            check_parameter_type(match_non_match_ratio, 'match_non_match_ratios items', tuple, 'tuple')

        np.random.seed(random_state)

        print("#MATCH-NON_MATCH SCENARIOS: {}".format(len(match_non_match_ratios)))
        print("MATCH-NON_MATCH SCENARIOS: {}".format(match_non_match_ratios))

        # change ground truth format: from triplet to expanded format
        gt_matching_pairs = dataset.copy()
        gt_matching_pairs_match = gt_matching_pairs[gt_matching_pairs["label"] == 1]
        gt_matching_pairs_non_match = gt_matching_pairs[gt_matching_pairs["label"] == 0]
        gt_data1_container, gt_data2_container, gt_integrated_dataset_container = create_integration_dataset_by_data_type(
            gt_matching_pairs, 'label')
        gt_data1 = gt_data1_container.get_data()
        gt_data2 = gt_data2_container.get_data()
        gt_integrated_dataset = gt_integrated_dataset_container.get_data()

        tokenized_gt_data1 = dataset_tokenizer(gt_data1, columns=columns)
        tokenized_gt_data2 = dataset_tokenizer(gt_data2, columns=columns)

        # get integrated matching/non-matching entities
        gt_integrated_dataset_match = gt_integrated_dataset.groupby('entity_id').filter(lambda x: len(x) > 1)
        gt_integrated_dataset_non_match = gt_integrated_dataset.groupby('entity_id').filter(lambda x: len(x) == 1)
        num_match_integrated_entities = len(gt_integrated_dataset_match['entity_id'].unique())
        num_non_match_integrated_entities = len(gt_integrated_dataset_non_match['entity_id'].unique())
        num_integrated_entities = gt_integrated_dataset['entity_id'].max()

        # get matching/non-matching entities for each data source
        # source 1
        # gt_data1[gt_data1['id'].isin(gt_integrated_dataset_match[gt_integrated_dataset_match['source'] == 0]['id'].values)]
        gt_data1_match_non_match_mask = gt_data1.apply(lambda x: gt_matching_pairs[
            (gt_matching_pairs['ltable_id'] == x['id']) & (gt_matching_pairs['label'] == 1)].empty, axis=1)
        gt_data1_match = gt_data1.loc[~gt_data1_match_non_match_mask, :]
        gt_data1_non_match = gt_data1.loc[gt_data1_match_non_match_mask, :]

        # source 2
        gt_data2_match_non_match_mask = gt_data2.apply(lambda x: gt_matching_pairs[
            (gt_matching_pairs['rtable_id'] == x['id']) & (gt_matching_pairs['label'] == 1)].empty, axis=1)
        gt_data2_match = gt_data2.loc[~gt_data2_match_non_match_mask, :]
        gt_data2_non_match = gt_data2.loc[gt_data2_match_non_match_mask, :]

        # create scenarios
        self.scenarios = []
        print("Creating scenarios...")

        # create a scenario for each user-provided correctly predicted match/non-match entity percentages
        for scenario_id, match_non_match_ratio in enumerate(match_non_match_ratios, 1):

            # get matching/non-matching ratios for the current scenario
            match_ratio = match_non_match_ratio[0]
            non_match_ratio = match_non_match_ratio[1]
            print("Scenario {} (match={}, non_match={})".format(scenario_id, match_ratio, non_match_ratio))
            variable_size_map = {}

            # create a new integrated dataset containing an entity matching error rate equals to match_ratio
            if match_ratio == 0:
                pred_integrated_dataset_match = gt_integrated_dataset_match.copy()
            else:
                # flip in the ground truth integrated dataset the entity ids of the matching entities that will not
                # classified correctly
                pred_integrated_dataset_match = gt_integrated_dataset_match.copy()

                # get a percentage of matching records (from the 2 data sources) equals to match_ratio
                data1_sample_match_ids = gt_data1_match.sample(frac=match_ratio, random_state=random_state)['id'].values
                data2_sample_match_ids = gt_data2_match.sample(frac=match_ratio, random_state=random_state)['id'].values
                variable_size_map['data1_wrong_match'] = len(data1_sample_match_ids)
                variable_size_map['data1_right_match'] = len(gt_data1_match) - len(data1_sample_match_ids)
                variable_size_map['data2_wrong_match'] = len(data2_sample_match_ids)
                variable_size_map['data2_right_match'] = len(gt_data2_match) - len(data2_sample_match_ids)
                variable_size_map['data1_data2_wrong_match'] = len(data1_sample_match_ids) + len(data2_sample_match_ids)
                variable_size_map['data1_data2_right_match'] = variable_size_map['data1_right_match'] + \
                                                               variable_size_map['data2_right_match']

                # create never seen entity ids and assign them to the considered matching entities (in this way they
                # will be recognized as isolated entities to be concatenated)
                pred_match_entity_ids = [i + num_integrated_entities + 1 for i in
                                         range(len(data1_sample_match_ids) + len(data2_sample_match_ids))]

                # select from the ground truth integrated dataset the matching entities to be wrongly classified and
                # assign them the new entity ids
                mask_data1_sample = (pred_integrated_dataset_match["source"] == 0) & (
                    pred_integrated_dataset_match['id'].isin(data1_sample_match_ids))
                mask_data2_sample = (pred_integrated_dataset_match["source"] == 1) & (
                    pred_integrated_dataset_match['id'].isin(data2_sample_match_ids))
                pred_integrated_dataset_match.loc[
                    mask_data1_sample | mask_data2_sample, 'entity_id'] = pred_match_entity_ids

            # create a new integrated dataset containing an entity non-matching error rate equals to non_match_ratio
            if non_match_ratio == 0:
                pred_integrated_dataset_non_match = gt_integrated_dataset_non_match.copy()
            else:
                # flip in the ground truth integrated dataset the entity ids of the non-matching entities that will not
                # classified correctly
                pred_integrated_dataset_non_match = gt_integrated_dataset_non_match.copy()

                # get a percentage of non-matching records (from the 2 data sources) equals to non_match_ratio
                data1_sample_non_match_ids = gt_data1_non_match.sample(frac=non_match_ratio, random_state=random_state)[
                    'id'].values
                data2_sample_non_match_ids = gt_data2_non_match.sample(frac=non_match_ratio, random_state=random_state)[
                    'id'].values
                variable_size_map['data1_wrong_non_match'] = len(data1_sample_non_match_ids)
                variable_size_map['data1_right_non_match'] = len(gt_data1_non_match) - len(data1_sample_non_match_ids)
                variable_size_map['data2_wrong_non_match'] = len(data2_sample_non_match_ids)
                variable_size_map['data2_right_non_match'] = len(gt_data2_non_match) - len(data2_sample_non_match_ids)
                variable_size_map['data1_data2_wrong_non_match'] = len(data1_sample_non_match_ids) + len(
                    data2_sample_non_match_ids)
                variable_size_map['data1_data2_right_non_match'] = variable_size_map['data1_right_non_match'] + \
                                                               variable_size_map['data2_right_non_match']

                # create new entity ids and assign them to the considered non-matching entities
                # randomly select the entity ids of the matching entities of the ground truth integrated dataset
                # in this way the non-matching entities will (wrongly) match with other entities
                pred_non_match_entity_ids = np.random.choice(gt_integrated_dataset_match['entity_id'].unique(),
                                                             len(data1_sample_non_match_ids) + len(
                                                                 data2_sample_non_match_ids))

                # select from the ground truth integrated dataset the non-matching entities to be wrongly classified and
                # assign them the new entity ids
                mask_data1_sample = (pred_integrated_dataset_non_match["source"] == 0) & (
                    pred_integrated_dataset_non_match['id'].isin(data1_sample_non_match_ids))
                mask_data2_sample = (pred_integrated_dataset_non_match["source"] == 1) & (
                    pred_integrated_dataset_non_match['id'].isin(data2_sample_non_match_ids))
                pred_integrated_dataset_non_match.loc[
                    mask_data1_sample | mask_data2_sample, 'entity_id'] = pred_non_match_entity_ids

            num_pred_match_integrated_entities = len(pred_integrated_dataset_match['entity_id'].unique())
            num_pred_non_match_integrated_entities = len(pred_integrated_dataset_non_match['entity_id'].unique())

            # create the final new integrated dataset by combining matching and non-matching entities
            pred_integrated_dataset_data = pd.concat([pred_integrated_dataset_match, pred_integrated_dataset_non_match])
            gt_integrated_dataset_container.set_data(pred_integrated_dataset_data)
            data_fusion_comp = DataFusionComponent(gt_integrated_dataset_container)
            # FIXME
            # pred_integrated_dataset = data_fusion_comp.select_random_attribute_values(random_state)
            pred_integrated_dataset = data_fusion_comp.select_random_records(random_state)

            tokenized_pred_integrated_dataset = dataset_tokenizer(pred_integrated_dataset.get_data(), columns=columns)

            # collect the scenario parameters
            scenario_params = {"scenario": scenario_id, "match_ratio": match_ratio, "non_match_ratio": non_match_ratio}

            # other params related to the scenario
            other_params = {'data1_match': len(gt_data1_match), 'data1_non_match': len(gt_data1_non_match),
                            'data2_match': len(gt_data2_match), 'data2_non_match': len(gt_data2_non_match),
                            'integration_match': len(gt_integrated_dataset_match),
                            'integration_non_match': len(gt_integrated_dataset_non_match),
                            'integration_math_pairs': len(gt_matching_pairs_match),
                            'integration_non_match_pairs': len(gt_matching_pairs_non_match),
                            'num_original_integrated_entities': num_integrated_entities,
                            'num_original_match_integrated_entities': num_match_integrated_entities,
                            'num_original_non_match_integrated_entities': num_non_match_integrated_entities,
                            'num_pred_integrated_entities': len(
                                pred_integrated_dataset.get_data()['entity_id'].unique()),
                            'num_pred_match_integrated_entities': num_pred_match_integrated_entities,
                            'num_pred_non_match_integrated_entities': num_pred_non_match_integrated_entities}
            other_params.update(variable_size_map)
            scenario_params.update(other_params)

            sources = [tokenized_gt_data1, tokenized_gt_data2]
            integrations = [tokenized_pred_integrated_dataset, tokenized_pred_integrated_dataset]

            # create the scenario object
            scenario_data = [
                {
                    'sources': sources,
                    'integrations': integrations,
                    'data_type': 'all',
                    'sources_ids': ['data1_all_{}_{}'.format(match_ratio, non_match_ratio),
                                    'data2_all_{}_{}.format(match_ratio, non_match_ratio)'.format(match_ratio,
                                                                                                  non_match_ratio)],
                    'integrations_ids': ['integration_all_{}_{}'.format(match_ratio, non_match_ratio),
                                         'integration_all_{}_{}'.format(match_ratio, non_match_ratio)]
                }
            ]
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)

            self.scenarios.append(scenario)

        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-match-percentages')

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """

        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("{} evaluation (match={}, non_match={}, {})".format(data_type, scenario_params['match_ratio'],
                                                                              scenario_params['non_match_ratio'],
                                                                              metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


class IntegratedEntitiesVariationEvaluator(object):
    """
    This class evaluates a metric over a set of data integration scenarios where variable percentages of entities are
    modified from the provided ground truth integrated dataset.
    """

    def __init__(self, dataset: pd.DataFrame, changed_entity_percentages: list, data_fusion_option: int,
                 columns: list = None, random_state: int = 24):
        """
        This method initializes the evaluator by generating, from the input ground truth integrated dataset, multiple
        data integration scenarios where variable percentages of its entities are modified.

        :param dataset: Pandas DataFrame containing the ground truth dataset from which the scenarios will be generated
        :param changed_entity_percentages: list of percentages of entities to modify from the ground truth
        :param data_fusion_option: the data fusion technique (0=select by source id; 1=random record selection; 2=random
                                   attribute value selection)
        :param columns: optional list of columns to consider in the data
        :param random_state: the seed for the random choices
        """

        print("\nINTEGRATED-ENTITIES-VARIATION EVALUATOR")

        # check input parameter data types
        check_parameter_type(dataset, 'dataset', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(changed_entity_percentages, 'changed_entity_percentages', list, 'list')
        check_parameter_type(data_fusion_option, 'data_fusion_option', int, 'integer')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        for changed_entity_ratio in changed_entity_percentages:
            check_parameter_type(changed_entity_ratio, 'changed_entity_percentages items', float, 'float')

        data_fusion_options = [0, 1, 2]
        if data_fusion_option not in data_fusion_options:
            raise ValueError("Wrong value for parameter data_fusion_option. Only the values {} are allowed.".format(
                data_fusion_options))

        np.random.seed(random_state)

        print("#CHANGED ENTITIES SCENARIOS: {}".format(len(changed_entity_percentages)))
        print("CHANGED ENTITIES SCENARIOS: {}".format(changed_entity_percentages))

        # change ground truth format: from triplet to extended format
        gt_matching_pairs = dataset.copy()
        gt_data1_container, gt_data2_container, gt_extended_integration_container = create_integration_dataset_by_data_type(
            gt_matching_pairs, 'label')
        gt_data1 = gt_data1_container.get_data()
        gt_data2 = gt_data2_container.get_data()
        gt_extended_integration = gt_extended_integration_container.get_data()

        # apply the data fusion to the extended integration
        data_fusion_comp = DataFusionComponent(gt_extended_integration_container)
        if data_fusion_option == 0:
            gt_integration = data_fusion_comp.select_records_by_source_id(0)
        elif data_fusion_option == 1:
            gt_integration = data_fusion_comp.select_random_records(random_state)
        else:
            gt_integration = data_fusion_comp.select_random_attribute_values(random_state)
        gt_integration = gt_integration.get_data()

        # tokenize the datasets
        tokenized_gt_data1 = dataset_tokenizer(gt_data1, columns=columns)
        tokenized_gt_data2 = dataset_tokenizer(gt_data2, columns=columns)
        tokenized_gt_integration = dataset_tokenizer(gt_integration, columns=columns)

        # get integrated matching/non-matching records
        gt_extended_integration_match = gt_extended_integration.groupby('entity_id').filter(lambda x: len(x) > 1)
        gt_match_entities = gt_extended_integration_match['entity_id'].unique()
        gt_num_match_entities = len(gt_match_entities)
        gt_extended_integration_non_match = gt_extended_integration.groupby('entity_id').filter(lambda x: len(x) == 1)
        gt_non_match_entities = gt_extended_integration_non_match['entity_id'].unique()
        gt_num_non_match_entities = len(gt_non_match_entities)
        num_integrated_entities = gt_extended_integration['entity_id'].max()

        # create scenarios
        self.scenarios = []
        print("Creating scenarios...")

        # create a scenario for each user-provided percentage of integrated entities to modify
        for scenario_id, changed_entity_ratio in enumerate(changed_entity_percentages, 1):

            print("Scenario {} (changed entity ratio={})".format(scenario_id, changed_entity_ratio))

            pred_integration = gt_integration.copy()

            if changed_entity_ratio == 0.0: # no changes to ground truth
                tokenized_pred_integration = tokenized_gt_integration

            else:   # change the ground truth

                # sample a number of integration entities equal to changed_entity_ratio
                changed_entities = pred_integration.sample(frac=changed_entity_ratio, random_state=random_state)

                # remove the 50% of the selected entities from the integration result and duplicate the remaining 50%
                entity_ids_to_remove = changed_entities.index.values[:int(len(changed_entities) / 2)]
                entities_to_duplicate = changed_entities.iloc[~changed_entities.index.isin(entity_ids_to_remove), :]

                pred_integration = pd.concat([pred_integration, entities_to_duplicate])
                pred_integration = pred_integration.iloc[~pred_integration.index.isin(entity_ids_to_remove), :]

                tokenized_pred_integration = dataset_tokenizer(pred_integration, columns=columns)

            # collect the scenario parameters
            scenario_params = {"scenario": scenario_id, "changed_entities": changed_entity_ratio}

            # other params related to the scenario
            other_params = {'gt_num_entities': num_integrated_entities,
                            'gt_num_match_entities': gt_num_match_entities,
                            'gt_num_non_match_entities': gt_num_non_match_entities,
                            'pred_num_entities': len(pred_integration['entity_id'].unique())}
            scenario_params.update(other_params)

            sources = [tokenized_gt_data1, tokenized_gt_data2]
            integrations = [tokenized_pred_integration, tokenized_pred_integration]

            # create the scenario object
            scenario_data = [
                {
                    'sources': sources,
                    'integrations': integrations,
                    'data_type': 'all',
                    'sources_ids': ['data1', 'data2'],
                    'integrations_ids': ['integration_{}_seed={}'.format(changed_entity_ratio, random_state),
                                         'integration_{}_seed={}'.format(changed_entity_ratio, random_state)]
                }
            ]
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)

            self.scenarios.append(scenario)

        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'entities-variation')

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """

        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("{} evaluation (change entity ratio={}, {})".format(data_type,
                                                                              scenario_params['changed_entities'],
                                                                              metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


class MinimalityTotalityEvaluator(object):
    """
    This class evaluates the minimality and totality properties with a provided metric over some datasets.
    """

    def __init__(self, source: pd.DataFrame, integration: pd.DataFrame, source_id: int, data_fusion_option: int,
                 columns: list = None, random_state: int = 24):
        """
        This method initializes the evaluator by creating from the input datasets two category of scenarios used to
        evaluate the totality and minimality properties. The first category is created by comparing the source dataset
        with a variable selection of the integration dataset. The second category is created by comparing the source
        dataset with multiple versions of the integration dataset obtained by concatenating a variable percentages of
        its records.
        A data fusion technique is applied to the provided integration dataset. Multiple data fusion options are
        available:
            option = 0: the data fusion selects only the matching records from the input data source;
            option = 1: the data fusion selects randomly the matching records;
            option = 2: the data fusion merges the matching records by randomly selecting the attribute values
        Furthermore the integration dataset has to include the columns 'source' and 'entity_id' with identify
        respectively the provenance (i.e., the data source of origin) of each record and the identifier of the referred
        real-world entity. Both the datasets have to include an 'id' column containing the identifier of each record.

        :param source: Pandas DataFrame containing the first dataset where the minimality and totality properties will
                       be evaluated
        :param integration: Pandas DataFrame containing the second dataset where the minimality and totality properties
                            will be evaluated
        :param source_id: the identifier of the source dataset
        :param data_fusion_option: the data fusion technique (0=select by source id; 1=random record selection; 2=random
                                   attribute value selection)
        :param columns: optional list of columns to consider in the data
        :param random_state: seed for random choices
        """

        print("\nMINIMALITY-TOTALITY EVALUATOR")

        # check input parameter data types and values
        check_parameter_type(source, 'source', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(integration, 'integration', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(source_id, 'source_id', int, 'integer')
        check_parameter_type(data_fusion_option, 'data_fusion_option', int, 'integer')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        data_fusion_options = [0, 1, 2]
        if data_fusion_option not in data_fusion_options:
            raise ValueError("Wrong value for parameter data_fusion_option. Only the values {} are allowed.".format(
                data_fusion_options))

        if columns is not None:
            check_cols_in_dataframe(source, columns)
            check_cols_in_dataframe(integration, columns)

        check_cols_in_dataframe(source, ['id'])
        check_cols_in_dataframe(integration, ['id', 'source', 'entity_id'])

        # filter the data source based on the records included in the integration dataset
        integration_source_data = integration[integration['source'] == source_id]
        source = source[source['id'].isin(integration_source_data["id"].values)]

        # select from the integration dataset only the matching records and the non-matching records deriving from the
        # considered data source
        non_match_data = integration.groupby('entity_id').filter(lambda x: len(x) == 1)
        source_non_match_data = non_match_data[non_match_data['source'] == source_id]
        match_data = integration.groupby('entity_id').filter(lambda x: len(x) > 1)
        integration_data = pd.concat([match_data, source_non_match_data])
        integration = IntegratedDataset(integration_data, "index", 'source', 'entity_id')

        # apply the data fusion
        data_fusion_comp = DataFusionComponent(integration)
        if data_fusion_option == 0:
            integration = data_fusion_comp.select_records_by_source_id(source_id).get_data()
        elif data_fusion_option == 1:
            integration = data_fusion_comp.select_random_records(random_state).get_data()
        elif data_fusion_option == 2:
            integration = data_fusion_comp.select_random_attribute_values(random_state).get_data()

        # optionally project the considered datasets over the provided columns
        if columns is not None:
            data1 = source[columns].copy()
            data2 = integration[columns].copy()
        else:
            data1 = source.copy()
            data2 = integration.copy()

        percentages = np.linspace(0.1, 1, 10)

        self.scenarios = []
        print("Creating scenarios...")

        # check the totality property by comparing the considered data sources with a variable selection of the
        # integrated dataset
        for percentage in percentages:

            # select from the integrated dataset the current percentage of records
            # the new integrated dataset will contain only these records
            sample_data2 = data2.sample(frac=percentage, random_state=random_state)

            scenario_params = {'data1': len(data1), 'data2': len(sample_data2), 'frac': percentage, 'dup': 0}
            print("Scenario {}".format(scenario_params))

            # tokenize the datasets
            tokenized_data1 = dataset_tokenizer(data1, columns=columns)
            tokenized_data2 = dataset_tokenizer(sample_data2, columns=columns)

            # create the scenario object
            scenario_data = [
                {
                    'sources': [tokenized_data1],
                    'integrations': [tokenized_data2],
                    'data_type': 'all',
                    'sources_ids': ['data1_seed={}'.format(random_state)],
                    'integrations_ids': ['data2_frac={}_seed={}'.format(percentage, random_state)]
                }
            ]
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)

            self.scenarios.append(scenario)

        # check the minimality property by comparing the considered data source with modified versions of the integrated
        # dataset obtained by duplicating a certain percentage of records
        for percentage in percentages:

            # select from the integrated dataset the current percentage of records
            # the new integrated dataset will correspond to the concatenation of the original integrated dataset with
            # these records
            sample_data2 = data2.sample(frac=percentage, random_state=random_state)
            concat_data2 = pd.concat([data2, sample_data2])

            scenario_params = {'data1': len(data1), 'data2': len(concat_data2), 'frac': 0, 'dup': percentage}
            print("Scenario {}".format(scenario_params))

            # tokenize the datasets
            tokenized_data1 = dataset_tokenizer(data1, columns=columns)
            tokenized_data2 = dataset_tokenizer(concat_data2, columns=columns)

            # create the scenario object
            scenario_data = [
                {
                    'sources': [tokenized_data1],
                    'integrations': [tokenized_data2],
                    'data_type': 'all',
                    'sources_ids': ['data1_seed={}'.format(random_state)],
                    'integrations_ids': ['data2_dup={}_seed={}'.format(percentage, random_state)]
                }
            ]
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)

            self.scenarios.append(scenario)

        # # validate the metric by comparing the considered data source with modified versions of the integrated
        # # dataset obtained by jointly selecting a specified percentage of records and duplicating an other percentage
        # # of records
        # for percentage in percentages:
        #     # select from the integrated dataset the current percentage of records
        #     # the new integrated dataset will correspond to the concatenation of the original integrated dataset with
        #     # these records
        #     sample_data2 = data2.sample(frac=percentage, random_state=random_state)
        #     concat_data2 = pd.concat([data2, sample_data2])
        #
        #     scenario_params = {'data1': len(data1), 'data2': len(concat_data2), 'frac': 0, 'dup': percentage}
        #     print("Scenario {}".format(scenario_params))
        #
        #     # tokenize the datasets
        #     tokenized_data1 = dataset_tokenizer(data1, columns=columns)
        #     tokenized_data2 = dataset_tokenizer(concat_data2, columns=columns)
        #
        #     # create the scenario object
        #     scenario_data = [
        #         {
        #             'sources': [tokenized_data1],
        #             'integrations': [tokenized_data2],
        #             'data_type': 'all',
        #             'sources_ids': ['data1_seed={}'.format(random_state)],
        #             'integrations_ids': ['data2_dup={}_seed={}'.format(percentage, random_state)]
        #         }
        #     ]
        #     scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
        #
        #     self.scenarios.append(scenario)

        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'minimality-totality')

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """

        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("{} evaluation ({})".format(data_type, metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


class RepresentativenessVarianceEvaluator(object):
    """
    This class evaluates the variance of the representativeness scores when computed over multiple data integration
    scenarios with variable sampled data.
    """

    def __init__(self, experiment_name: str, experiment_params: dict, num_experiments: int, random_state: int = 24):
        """
        This method initializes the evaluator by generating multiple scenarios each one corresponding to the execution
        of an instance of the specified experiment over variable sampled data.

        :param experiment_name: the name of the experiment for which to evaluate the statistical significance of its
                                results
        :param experiment_params: the parameters of the experiment
        :param num_experiments: the number of times to repeat the experiment
        :param random_state: seed for random choices
        """

        print("\nEXPERIMENT-STATISTICAL-SIGNIFICANCE EVALUATOR")

        check_parameter_type(experiment_name, 'experiment_name', str, 'string')
        check_parameter_type(experiment_params, 'experiment_params', dict, 'dictionary')
        check_parameter_type(num_experiments, 'num_experiments', int, 'integer')
        check_parameter_type(random_state, 'random_state', int, 'integer')

        experiment_names = ['minimality-totality', 'entities-variation']
        if experiment_name not in experiment_names:
            raise ValueError(
                "Wrong value for parameter experiment_name. Only the values {} are allowed.".format(experiment_names))

        params = None
        if experiment_name == 'minimality-totality':
            params = ['source', 'integration', 'source_id', 'data_fusion_option', 'columns']

        elif experiment_name == 'entities-variation':
            params = ['dataset', 'changed_entity_percentages', 'data_fusion_option', 'columns']

        for param in params:
            if param not in experiment_params:
                raise ValueError(
                    "Missing parameter for the experiment {}. Param {} is required.".format(experiment_name, param))

        self.evaluators = []
        print("Creating scenarios...")

        random_states = range(1, num_experiments + 1)
        for ix, random_state in enumerate(random_states, 1):
            print("Preparing the experiment #{}".format(ix))

            evaluator = None
            if experiment_name == 'minimality-totality':
                source = experiment_params['source']
                integration = experiment_params['integration']
                source_id = experiment_params['source_id']
                data_fusion_option = experiment_params['data_fusion_option']
                columns = experiment_params['columns']
                evaluator = MinimalityTotalityEvaluator(source, integration, source_id, data_fusion_option,
                                                        columns=columns, random_state=random_state)

            elif experiment_name == 'entities-variation':
                dataset = experiment_params['dataset']
                changed_entity_percentages = experiment_params['changed_entity_percentages']
                data_fusion_option = experiment_params['data_fusion_option']
                columns = experiment_params['columns']
                evaluator = IntegratedEntitiesVariationEvaluator(dataset, changed_entity_percentages,
                                                                 data_fusion_option, columns=columns,
                                                                 random_state=random_state)

            self.evaluators.append(evaluator)

        self.experiment_name = experiment_name
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'experiment-significance',
                                        experiment_name)

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the evaluators
        results = []
        for ix, evaluator in enumerate(self.evaluators, 1):
            res = evaluator.evaluate_metric(metric_calculator, metric_params, mean)
            results.append(res)

        results = pd.concat(results)
        results = results.fillna('None')
        conf_params = ['mode', 'ngram', 'embed_manager']
        if self.experiment_name == 'minimality-totality':
            conf_params += ['frac', 'dup']
        elif self.experiment_name == 'entities-variation':
            conf_params += ['changed_entities']

        group_res_by_conf = results.groupby(conf_params)

        aggregated_results = []
        for conf_val, res_by_conf in group_res_by_conf:

            out_data = res_by_conf.iloc[0, :].copy()

            features = ['Ix', 'Iy', 'Iy normalized', 'score (x,y)', 'score (x,y norm)', 'norm score (x, y)',
                        'norm score (x, y norm)']
            for feature in features:
                out_data["{} mean".format(feature)] = np.mean(res_by_conf[feature].values)
                out_data["{} std".format(feature)] = np.std(res_by_conf[feature].values)

            confidence_intervals_map = {}
            for feature in features:
                conf_ints = []
                for i in range(1, 4):
                    conf_ints.append((out_data["{} mean".format(feature)] - i * out_data["{} std".format(feature)],
                                      out_data["{} mean".format(feature)] + i * out_data["{} std".format(feature)]))
                confidence_intervals_map[feature] = conf_ints

            def check_val_in_interval(val, interval):
                if interval[0] <= val <= interval[1]:
                    return 1
                return 0

            for feature in features:
                conf_ints = confidence_intervals_map[feature]
                for int_id, conf_int in enumerate(conf_ints, 1):
                    presence_in_interval = res_by_conf.apply(lambda x: check_val_in_interval(x[feature], conf_int),
                                                             axis=1)
                    out_data["{} interval{}".format(feature, int_id)] = np.sum(presence_in_interval) / len(
                        presence_in_interval)

            aggregated_results.append(out_data)

        self.results = pd.DataFrame(aggregated_results)

        return self.results
    
    
class RepresentativenessOverSamplesEvaluator(object):
    """
    This class evaluates some representativeness metrics over multiple data integration scenarios with variable sampled
    data.
    """
    # FIXME: if the columns parameter is not provided it is needed to remove some columns from the data fusion results
    def __init__(self, extended_integrated_dataset: pd.DataFrame, sample_sizes: list, columns: list = None,
                 random_state: int = 24):
        """
        This method initializes the evaluator by generating multiple scenarios obtained by sampling variable percentages
        of data from the provided integrated dataset.

        :param extended_integrated_dataset: Pandas DataFrame containing the integrated dataset where the scenarios will
                                            be generated
        :param sample_sizes: dimension of the data samples to consider
        :param random_state: seed for random choices
        """

        print("\nREPRESENTATIVENESS-OVER-SAMPLES EVALUATOR")

        check_parameter_type(extended_integrated_dataset, 'extended_integrated_dataset', pd.DataFrame,
                             'Pandas DataFrame')
        check_parameter_type(sample_sizes, 'sample_sizes', list, 'list')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        integration_cols = ['index', 'source', 'entity_id']
        check_cols_in_dataframe(extended_integrated_dataset, integration_cols)

        if len(sample_sizes) == 0:
            raise ValueError("Empty list of sample sizes provided.")

        for ss in sample_sizes:
            check_parameter_type(ss, 'single sample size', float, 'float')

        if columns is not None:
            check_cols_in_dataframe(extended_integrated_dataset, columns)

        # get matching and non-matching entities in the integrated dataset
        tot_entities =  extended_integrated_dataset['entity_id'].max()
        matching_records = extended_integrated_dataset.groupby('entity_id').filter(lambda x: len(x) > 1)
        matching_entities = pd.Series(matching_records['entity_id'].unique())
        match_ratio = len(matching_entities) / tot_entities
        non_matching_records = extended_integrated_dataset.groupby('entity_id').filter(
            lambda x: len(x) == 1)
        non_matching_entities = pd.Series(non_matching_records['entity_id'].unique())
        non_match_ratio = len(non_matching_entities) / tot_entities

        # extract from the integrated dataset the data related to the single data sources
        data1 = extended_integrated_dataset[extended_integrated_dataset['source'] == 0]
        data2 = extended_integrated_dataset[extended_integrated_dataset['source'] == 1]

        # apply the data fusion to the input integrated dataset
        integrated_dataset_container = IntegratedDataset(extended_integrated_dataset, 'index', 'source',
                                                         'entity_id')
        data_fusion_comp = DataFusionComponent(integrated_dataset_container)
        original_integrated_dataset = data_fusion_comp.select_random_attribute_values(24).get_data()

        common_params = {'data1': len(data1), 'data2': len(data2), 'ext_integration': len(extended_integrated_dataset),
                         'integration': len(original_integrated_dataset)}
        scenario_params = {'scenario': 0, 'sample1': len(data1), 'sample2': len(data2),
                           'sample_integration': len(original_integrated_dataset), 'sample_size': 1.0}
        scenario_params.update(common_params)
        print("Target scenario {}".format(scenario_params))

        # tokenize the datasets
        tokenized_data1 = dataset_tokenizer(data1, columns=columns)
        tokenized_data2 = dataset_tokenizer(data2, columns=columns)
        tokenized_integration = dataset_tokenizer(original_integrated_dataset, columns=columns)

        # create the scenario object
        scenario_data = [
            {
                'sources': [tokenized_data1, tokenized_data2],
                'integrations': [tokenized_integration, tokenized_integration],
                'data_type': 'all',
                'sources_ids': ['data1', 'data2'],
                'integrations_ids': ['integration', 'integration']
            }
        ]
        target_scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
        scenarios = [target_scenario]

        print("Creating scenarios...")

        # create a sample of the integrated dataset for each sample size
        for sc, sample_size in enumerate(sample_sizes, 1):

            # extract from the integrated dataset sample_size / 2 matching and non matching entities
            match_sample_size = int(round(match_ratio * sample_size * tot_entities))
            non_match_sample_size = int(round(non_match_ratio * sample_size * tot_entities))
            match_sample = matching_entities.sample(n=match_sample_size, random_state=random_state)
            non_match_sample = non_matching_entities.sample(n=non_match_sample_size, random_state=random_state)
            sampled_entities = list(match_sample.values) + list(non_match_sample.values)
            sample_ext_integration = extended_integrated_dataset[
                extended_integrated_dataset['entity_id'].isin(sampled_entities)]
            integration = original_integrated_dataset[
                original_integrated_dataset['entity_id'].isin(sampled_entities)]

            # extract from the considered portion of the integrated dataset, the data of the two data sources
            s1 = sample_ext_integration[sample_ext_integration['source'] == 0]
            s2 = sample_ext_integration[sample_ext_integration['source'] == 1]

            scenario_params = {'scenario': sc, 'sample1': len(s1), 'sample2': len(s2),
                               'sample_integration': len(integration), 'sample_size': sample_size}
            scenario_params.update(common_params)
            print("scenario {}".format(scenario_params))

            # tokenize the datasets
            tokenized_s1 = dataset_tokenizer(s1, columns=columns)
            tokenized_s2 = dataset_tokenizer(s2, columns=columns)
            tokenized_int = dataset_tokenizer(integration, columns=columns)

            # create the scenario object
            scenario_data = [
                {
                    'sources': [tokenized_s1, tokenized_s2],
                    'integrations': [tokenized_int, tokenized_int],
                    'data_type': 'all',
                    'sources_ids': ['data{}_seed={}_size={}'.format(i, random_state, sample_size) for i in range(1, 3)],
                    'integrations_ids': ['int_seed={}_size={}'.format(random_state, sample_size) for _ in range(1, 3)]
                }
            ]
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
            scenarios.append(scenario)

        self.scenarios = scenarios
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'sampling')

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("{} evaluation ({})".format(data_type, metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


class RepresentativenessBiasEvaluator(object):
    """
    This class evaluates the bias of the representativeness scores when computed over multiple data integration
    scenarios with variable sampled data.
    """

    def __init__(self, experiment_name: str, experiment_params: dict, num_experiments: int, random_state: int = 24):
        """
        This method initializes the evaluator by generating multiple scenarios each one corresponding to the execution
        of an instance of the specified experiment over variable sampled data.

        :param experiment_name: the name of the experiment for which to evaluate the statistical significance of its
                                results
        :param experiment_params: the parameters of the experiment
        :param num_experiments: the number of times to repeat the experiment
        :param random_state: seed for random choices
        """

        print("\nREPRESENTATIVENESS-BIAS EVALUATOR")

        check_parameter_type(experiment_name, 'experiment_name', str, 'string')
        check_parameter_type(experiment_params, 'experiment_params', dict, 'dictionary')
        check_parameter_type(num_experiments, 'num_experiments', int, 'integer')
        check_parameter_type(random_state, 'random_state', int, 'integer')

        experiment_names = ['bias']
        if experiment_name not in experiment_names:
            raise ValueError(
                "Wrong value for parameter experiment_name. Only the values {} are allowed.".format(
                    experiment_names))

        params = None
        if experiment_name == 'bias':
            params = ['extended_integrated_dataset', 'sample_sizes', 'columns']

        for param in params:
            if param not in experiment_params:
                raise ValueError(
                    "Missing parameter for the experiment {}. Param {} is required.".format(experiment_name, param))

        self.evaluators = []
        print("Creating scenarios...")

        seeds = range(1, num_experiments + 1)
        for ix, seed in enumerate(seeds, 1):
            print("Preparing the experiment #{}".format(ix))

            evaluator = None
            if experiment_name == 'bias':
                integration = experiment_params['extended_integrated_dataset']
                sample_sizes = experiment_params['sample_sizes']
                columns = experiment_params['columns']
                evaluator = RepresentativenessOverSamplesEvaluator(integration, sample_sizes, columns=columns,
                                                                   random_state=seed)

            self.evaluators.append(evaluator)

        self.experiment_name = experiment_name
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'bias', experiment_name)

        print("Scenarios created successfully.")

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator,
                             'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the evaluators
        results = []
        for ix, evaluator in enumerate(self.evaluators, 1):
            res = evaluator.evaluate_metric(metric_calculator, metric_params, mean)
            results.append(res)

        results = pd.concat(results)
        results = results.fillna('None')

        target_rows = results[results['scenario'] == 0]
        results = results.iloc[~results.index.isin(target_rows.index.values), :]

        conf_params = ['mode', 'ngram', 'embed_manager']
        if self.experiment_name == 'bias':
            conf_params += ['sample_size']

        group_res_by_conf = results.groupby(conf_params)

        aggregated_results = []
        for conf_val, res_by_conf in group_res_by_conf:

            out_data = res_by_conf.iloc[0, :].copy()

            features = ['Ix', 'Iy', 'Iy normalized', 'score (x,y)', 'score (x,y norm)', 'norm score (x, y)',
                        'norm score (x, y norm)']
            for feature in features:

                diff_metrics = get_regression_metric_scores(target_rows[feature].values, res_by_conf[feature].values)

                for key in diff_metrics:
                    out_data["{}({})".format(key, feature)] = diff_metrics[key]

            aggregated_results.append(out_data)

        self.results = pd.DataFrame(aggregated_results)

        return self.results


class TimePerformanceEvaluator(object):
    """
    This class evaluates the time performance of a metric in computing the representativeness scores over a data
    integration scenario.
    """

    def __init__(self, extended_integrated_dataset: pd.DataFrame = None, datasets_map: dict = None,
                 columns: list = None, random_state: int = 24):
        """
        This method initializes the evaluator by creating an integration scenario from the input datasets. The user can
        provide an extended integrated dataset or a map of already-prepared (data sources, integrated datasets).
        In the first case, from the extended integrated dataset will be extracted some data sources which will
        be compared with the single integrated dataset.
        In the second case, the provided data sources will be compared with the multiple integrated datasets.

        :param extended_integrated_dataset: optional extended integrated dataset from which some data sources will be
                                            extracted
        :param datasets_map: optional map of already-prepared datasets (split in data sources and integrated datasets)
        :param columns: optional list of columns to consider in the data
        :param random_state: the seed for the random choices
        """

        print("\nTIME PERFORMANCE EVALUATOR")

        # check input parameter data types
        check_parameter_type(extended_integrated_dataset, 'extended_integrated_dataset', pd.DataFrame,
                             'Pandas DataFrame', optional_param=True)
        check_parameter_type(datasets_map, 'datasets_map', dict, 'dictionary', optional_param=True)
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        # check input parameter data values
        if extended_integrated_dataset is not None:

            cols = ['index', 'id', 'source', 'entity_id']
            check_cols_in_dataframe(extended_integrated_dataset, cols)

            if columns is not None:
                check_cols_in_dataframe(extended_integrated_dataset, columns)

        if datasets_map is not None:
            if 'sources' not in datasets_map or 'integrations' not in datasets_map:
                msg_err = "Wrong data format for parameter datasets_map."
                msg_err += " The dictionary has to contain the keys 'sources' and 'integrations'."
                raise ValueError(msg_err)
            sources = datasets_map['sources']
            integrations = datasets_map['integrations']
            check_parameter_type(sources, 'datasets map sources list', list, 'list')
            check_parameter_type(integrations, 'datasets map integrations list', list, 'list')

            for d in sources:
                check_parameter_type(d, 'datasets map single source', pd.DataFrame, 'Pandas DataFrame')
                if columns is not None:
                    check_cols_in_dataframe(d, columns)

            for d in integrations:
                check_parameter_type(d, 'datasets map single integrated dataset', pd.DataFrame, 'Pandas DataFrame')
                if columns is not None:
                    check_cols_in_dataframe(d, columns)

        print("\nCreating scenario...")

        scenarios = []
        if extended_integrated_dataset is not None:

            # extract from the provided integrated dataset some data source datasets
            original_sources = []
            sources_ids = extended_integrated_dataset['source'].unique()
            for source_id in sources_ids:
                source = extended_integrated_dataset[extended_integrated_dataset['source'] == source_id]
                del source['index']
                del source['source']
                del source['entity_id']
                original_sources.append(source)

            # apply the data fusion to the input integrated dataset
            integrated_dataset_container = IntegratedDataset(extended_integrated_dataset, 'index', 'source',
                                                             'entity_id')
            data_fusion_comp = DataFusionComponent(integrated_dataset_container)
            original_integrated_dataset = data_fusion_comp.select_random_attribute_values(random_state).get_data()

            # select variable percentages of data from the original data sources and integrated dataset
            data_sizes = [1000, 10000, 50000, 100000, 500000, 1000000]
            for scenario_id, data_size in enumerate(data_sizes, 1):

                print("Creating scenario with size: {}".format(data_size))

                sources = []
                for original_source in original_sources:
                    source = original_source.sample(n=data_size, random_state=random_state, replace=True)
                    sources.append(source)

                integrated_dataset = original_integrated_dataset.sample(n=data_size, random_state=random_state,
                                                                        replace=True)

                integrations = [integrated_dataset, integrated_dataset]
                sources_ids = ["s{}_sc={}".format(x, scenario_id) for x in range(1, len(sources) + 1)]
                integrations_ids = ["i_sc={}".format(scenario_id), "i_sc={}".format(scenario_id)]

                tokenized_sources = [dataset_tokenizer(d, columns=columns) for d in sources]
                tokenized_integrations = [dataset_tokenizer(d, columns=columns) for d in integrations]

                scenario_data = [
                    {
                        'sources': tokenized_sources,
                        'integrations': tokenized_integrations,
                        'data_type': 'all',
                        'sources_ids': sources_ids,
                        'integrations_ids': integrations_ids
                    }
                ]
                scenario_params = {'scenario': scenario_id, 'data_size': data_size}
                scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
                scenarios.append(scenario)


        if datasets_map is not None:
            sources = datasets_map['sources']
            integrations = datasets_map['integrations']
            sources_ids = ["s_{}".format(x) for x in range(1, len(sources) + 1)]
            # integrations_ids = ["i_{}".format(x) for x in range(1, len(integrations) + 1)]
            integrations_ids = ["i" for x in range(1, len(integrations) + 1)]

            tokenized_sources = [dataset_tokenizer(d, columns=columns) for d in sources]
            tokenized_integrations = [dataset_tokenizer(d, columns=columns) for d in integrations]

            scenario_data = [
                {
                    'sources': tokenized_sources,
                    'integrations': tokenized_integrations,
                    'data_type': 'all',
                    'sources_ids': sources_ids,
                    'integrations_ids': integrations_ids
                }
            ]
            scenario_params = {'scenario': 1}

            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
            scenarios.append(scenario)

        print("\nScenarios created successfully.")

        self.scenarios = scenarios
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'time-performance')

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """
        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            # scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                extra_results = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("\n{} evaluation ({})".format(data_type, metric_param))

                    # calculate the metric score
                    start_time = time.time()
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)
                    end_time = time.time() - start_time

                    scenario_scores.append(scenario_score)
                    extra_results.append({'total_time': end_time})

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources),
                                            extra_results=extra_results)

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results


class MultiSourceIntegrationEvaluator(object):
    """
    This class evaluates some representativeness measures over a multi-source data integration scenario.
    """

    def __init__(self, sources: list, num_sources: int, columns: list = None, random_state: int = 24):
        """
        This method initializes the evaluator by creating multiple scenarios that will operate on a multi-source data
        integration scenario.

        :param sources: the list of the data sources given in input to the integration process
        :param num_sources: number of data sources to consider (from the provided list of data sources)
        :param columns: optional list of columns to consider in the data
        :param random_state: the seed for the random choices
        """

        print("\nMULTI-SOURCE INTEGRATION EVALUATOR")

        # check input parameter data types
        check_parameter_type(sources, 'sources', list, 'list')
        check_parameter_type(num_sources, 'num_sources', int, 'int')
        check_parameter_type(columns, 'columns', list, 'list', optional_param=True)
        check_parameter_type(random_state, 'random_state', int, 'integer')

        # check input parameter data values
        for s in sources:
            check_parameter_type(s, 'single source', pd.DataFrame, 'Pandas DataFrame')

        if len(sources) < num_sources:
            raise ValueError(
                "Wrong value for parameter num_sources. Only values smaller than {} are allowed.".format(len(sources)))

        if columns is not None:
            for s in sources:
                check_cols_in_dataframe(s, columns)

        # select only <num_sources> data sources
        sources = [sources[i] for i in range(num_sources)]

        # create multiple data integration scenarios
        print("\nCreating scenario...")

        scenarios = []
        for i in range(num_sources):

            integration = sources[i].copy()

            integrations = [integration] * num_sources
            sources_ids = ["s{}".format(x) for x in range(num_sources)]
            integrations_ids = ["s{}".format(i)] * num_sources

            tokenized_sources = [dataset_tokenizer(d, columns=columns) for d in sources]
            tokenized_integrations = [dataset_tokenizer(d, columns=columns) for d in integrations]

            scenario_data = [
                {
                    'sources': tokenized_sources,
                    'integrations': tokenized_integrations,
                    'data_type': 'all',
                    'sources_ids': sources_ids,
                    'integrations_ids': integrations_ids
                }
            ]
            scenario_params = {'scenario': i + 1}
            scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)
            scenarios.append(scenario)

            # break

        print("\nScenarios created successfully.")

        self.scenarios = scenarios
        self.results = None
        self.results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-source-integration')

    def save_results(self, file_prefix: str):
        """
        This method saves the results to disk.

        :param file_prefix: the prefix to be added to result file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.results_dir)

        file_name = os.path.join(self.results_dir, "{}_results.csv".format(file_prefix))
        self.results.to_csv(file_name, index=False)

    def get_scenarios(self):
        """
        This method returns the generated scenarios.

        :return: list of scenario objects
        """
        return self.scenarios

    def evaluate_metric(self, metric_calculator: TwoWayMetricCalculator, metric_params: list, mean: bool = False):
        """
        This method evaluates the user-provided metric over the generated scenarios.

        :param metric_calculator: the metric to evaluate
        :param metric_params: a list of metric configurations
        :param mean: a boolean flag that indicates whether to computes also aggregated metric scores
        :return: Pandas DataFrame containing the results of the evaluation
        """

        # check input parameters
        check_parameter_type(metric_calculator, 'metric_calculator', TwoWayMetricCalculator, 'TwoWayMetricCalculator')
        check_parameter_type(metric_params, 'metric_params', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        # loop over the scenarios
        for i, scenario in enumerate(self.scenarios, 1):
            print()
            print('---- Scenario {} ----'.format(i))
            print()

            scenario_params = scenario.get_params()
            scenario_data = scenario.get_data()

            # loop over the multiple metric configurations
            for metric_param in metric_params:

                # evaluate the metric over multiple data types
                scenario_scores = []
                for scenario_item in scenario_data:
                    sources = scenario_item['sources']
                    integrations = scenario_item['integrations']
                    data_type = scenario_item['data_type']
                    sources_ids = scenario_item['sources_ids']
                    integrations_ids = scenario_item['integrations_ids']

                    print("{} evaluation ({})".format(data_type, metric_param))

                    # calculate the metric score
                    scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations,
                                                                                      sources_ids, integrations_ids,
                                                                                      metric_param, mean)

                    scenario_scores.append(scenario_score)

                # save scenario scores
                export_metric_param = metric_calculator.__class__.convert_params_for_export(metric_param)
                scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

        results = []
        for scenario in self.scenarios:
            all_scores = scenario.get_scores()
            for score_key in all_scores:
                results += all_scores[score_key]

        self.results = pd.DataFrame(results)

        return self.results
