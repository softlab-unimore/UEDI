import collections
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from uedi.models.data_models import DataSource, IntegratedDataset
from uedi.data_integration.data_fusion import DataFusionComponent
from uedi.utils.general_utilities import check_parameter_type


# OLD PART
class AdvancedIntegrationScenariosGeneration(object):
    """
    This class manages the generation of integration scenarios.
    """

    def __init__(self, sources, integrations):
        """
        This method initializes the variables needed for creating integration scenarios.

        :param sources: list of data sources
        :param integrations: list of integration results obtained by integrating the user-provided data sources
        """

        if not isinstance(sources, collections.Iterable):
            raise TypeError("Wrong data type for parameter sources. Only iterable data type is allowed.")

        if not isinstance(integrations, collections.Iterable):
            raise TypeError("Wrong data type for parameter integrations. Only iterable data type is allowed.")

        for source in list(sources):
            if not isinstance(source, DataSource):
                raise TypeError(
                    "Wrong data type for the elements of the sources parameter. Only DataSource data type is allowed.")

        for integration in list(integrations):
            if not isinstance(integration, IntegratedDataset):
                raise TypeError(
                    "Wrong data type for integrations parameter elements. Only IntegratedDataset data type is allowed.")

        self.sources = list(sources)
        self.integrations = list(integrations)

    def get_perfect_scenario(self):
        """
        This method returns the perfect scenario: the one provided in input by the user when instantiating this class.
        :return: (list of data sources, list of their integration results)
        """
        return self.sources, self.integrations

    def generate_wrong_scenario_by_randomly_flipping(self, integration_ids, seed, num_flips=None, error_rates=None,
                                                     supports=None, debug=True):
        """
        This method modifies the perfect scenario by randomly flipping values in the user-specified integrations.
        For each integration the user wants to modify, he/she has to indicate the number of values to be flipped (or
        the error_ratio that indicates the number of flips with respect the integrated dataset size) and
        optionally the support used to select the entities to be changed.

        :param integration_ids: the indexes of the integrations to be modified
        :param seed: the seed for the random choices
        :param num_flips: the number of values to be flipped for each integration
        :param error_rates: the ratios of the integrated dataset to be considered as the number of flips
        :param supports: the minimum number of records per entity, used to select the entities to be changed
        :param debug: flag for enabling the debug mode
        :return: (list of original sources, list of modified integrations)
        """

        # check data types
        if not isinstance(integration_ids, collections.Iterable):
            raise TypeError("Wrong data type for parameter integration_ids. Only iterable data type is allowed.")

        for integration_id in integration_ids:
            if not isinstance(integration_id, int):
                raise TypeError("Wrong data type for integration_ids elements. Only integer data type is allowed.")

        if not isinstance(seed, int):
            raise TypeError("Wrong data type for parameter seed. Only integer data type is allowed.")

        if num_flips is not None:

            if not isinstance(num_flips, collections.Iterable):
                raise TypeError("Wrong data type for parameter num_flips. Only iterable data type is allowed.")

            for num_flip in num_flips:
                if not isinstance(num_flip, int):
                    raise TypeError("Wrong data type for num_flips elements. Only integer data type is allowed.")

        if error_rates is not None:

            if not isinstance(error_rates, collections.Iterable):
                raise TypeError("Wrong data type for parameter error_rates. Only iterable data type is allowed.")

            for error_rate in error_rates:
                if not isinstance(error_rate, float):
                    raise TypeError("Wrong data type for error_rates elements. Only float data type is allowed.")

        if supports is not None:
            if not isinstance(supports, collections.Iterable):
                raise TypeError("Wrong data type for parameter supports. Only iterable data type is allowed.")

            for support in supports:
                if not isinstance(support, int):
                    raise TypeError("Wrong data type for supports elements. Only integer data type is allowed.")

        # check data values
        num_integrations = len(self.integrations)
        for integration_id in integration_ids:
            if integration_id < 0 or integration_id >= num_integrations:
                raise ValueError(
                    "Wrong value for integration_ids elements. Only values in the range [0, {}) are allowed.".format(
                        num_integrations))

        # num_splits and supports values are checked by the function flip_randomly_entity_id
        if num_flips is not None and error_rates is not None:
            raise ValueError("Provide only one of the following parameters: num_flips, error_rates (not both).")

        num_error_list = None
        if num_flips is not None:
            num_error_list = num_flips
        if error_rates is not None:
            num_error_list = error_rates
        same_length_lists = [integration_ids, num_error_list]

        if supports is not None:
            same_length_lists.append(supports)

        if len(np.unique([len(vector) for vector in same_length_lists])) > 1:
            raise ValueError("User-provided lists have not the same lengths.")

        integration_ids = list(integration_ids)

        out_integrations = self.integrations[:]
        for i in range(len(integration_ids)):
            integration_id = integration_ids[i]
            if num_flips is not None:
                num_flip = list(num_flips)[i]
            else:
                num_flip = None
            if error_rates is not None:
                error_rate = list(error_rates)[i]
            else:
                error_rate = None

            if supports is not None:
                support = list(supports)[i]
            else:
                support = 1

            integration = self.integrations[integration_id]
            out_integrations[integration_id], _, _ = integration.flip_randomly_entity_id(num_flip=num_flip, seed=seed,
                                                                                         error_rate=error_rate,
                                                                                         support=support,
                                                                                         debug=debug)

        return self.sources, out_integrations


class SimpleIntegrationScenariosGeneration(object):
    """
    This class manages the creation of simple integration scenarios.
    """

    def __init__(self, sources, integrations):
        """
        This method stores the user-provided sources and integrated datasets. This data will be used to create multiple
        simple integration scenarios.

        :param sources: list of DataSource objects containing the data source
        :param integrations: list of IntegratedDataset objects containing the integrated datasets
        """

        if not isinstance(sources, collections.Iterable):
            raise TypeError("Wrong data type for parameter sources. Only iterable data type is allowed.")

        if not isinstance(integrations, collections.Iterable):
            raise TypeError("Wrong data type for parameter integrations. Only iterable data type is allowed.")

        for source in sources:
            if not isinstance(source, DataSource):
                raise TypeError("Wrong data type for sources elements. Only DataSource data type is allowed.")

        for integration in integrations:
            if not isinstance(integration, IntegratedDataset):
                raise TypeError(
                    "Wrong data type for integrations elements. Only IntegratedDataset data type is allowed.")

        sources = list(sources)
        integrations = list(integrations)

        if len(sources) != 3 and len(integrations) != 2:
            raise ValueError(
                "For this simple scenario generator class only 3 data sources and 2 integrations are allowed.")

        self.sources = sources
        self.integrations = integrations
        self.random_seed = 24

        print("SOURCES")
        for i, source in enumerate(sources):
            print(source.get_data().shape)

    @staticmethod
    def _create_integrated_dataset_from_data_source(data_source):

        integration = data_source.get_data().copy()
        entity_ids = range(len(integration))
        integration["entity_id"] = entity_ids
        integration["source_id"] = data_source.get_source_id()
        integration.reset_index(inplace=True)
        integrated_dataset = IntegratedDataset(integration, "index", "source_id", "entity_id")

        return integrated_dataset

    def get_scenario1_match(self):
        """
        This method creates the scenario 'scenario1_match' which consists into 3 data sources with the same content
        and two integrated datasets which contain the same contents of the data sources.
        :return: (a list of data sources, a list of integrated datasets)
        """

        data1 = self.sources[0]
        perfect_integration_dataset = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(
            data1)

        out_sources = [data1.copy(), data1.copy(), data1.copy()]
        out_integrations = [perfect_integration_dataset, perfect_integration_dataset.copy()]

        return out_sources, out_integrations

    def get_scenario1_no_match(self):
        """
        This method creates the scenario 'scenario1_no_match' which consists into 3 data sources with the same
        content and two integrated datasets. The first one contains the same contents of the data sources, while the
        second one is obtained by concatenating the contents of the data sources (i.e., no integration is applied).
        :return: (a list of data sources, a list of integrated datasets)
        """

        data1 = self.sources[0]
        first_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(data1)
        concat_data = DataSource(pd.concat([data1.get_data(), data1.get_data()]), 1)
        second_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(
            concat_data)

        out_sources = [data1.copy(), data1.copy(), data1.copy()]
        out_integrations = [first_integration, second_integration]

        return out_sources, out_integrations

    def get_scenario1(self):
        """
        This method creates the first scenario which consists in two sub-scenarios, where the integration is performed
        correctly and no integration is applied.
        :return: (list of sources, list of correctly integrated datasets, list of non integrated datasets)
        """

        sources, perfect_integrations = self.get_scenario1_match()
        _, wrong_integrations = self.get_scenario1_no_match()

        return sources, perfect_integrations, wrong_integrations

    def get_scenario2_match(self):
        """
        This method creates the scenario 'scenario2_match' which consists into 3 data sources, where the first two are
        equal and the third one is different, and two integrated datasets. The first integrated dataset contains the
        same content of the first two data sources, while the second one is obtained by itegrating the first (or the
        second) and the third data sources.
        :return: (a list of data sources, a list of integrated datasets)
        """

        data1 = self.sources[0]
        data2 = self.sources[1]
        integration = self.integrations[0]

        first_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(data1)

        second_integration = DataFusionComponent(integration).select_random_records(self.random_seed)

        out_sources = [data1.copy(), data1.copy(), data2.copy()]
        out_integrations = [first_integration, second_integration]

        return out_sources, out_integrations

    def get_scenario2_no_match(self):
        """
        This method creates the scenario 'scenario2_no_match' which consists into 3 data sources, where the first two
        are equal and the third one is different, and two integrated datasets. The first integrated dataset contains the
        same content of the first two data sources, while the second one is obtained by concatenating the first (or the
        second) and the third data sources (i.e., no integration is applied).
        :return: (a list of data sources, a list of integrated datasets)
        """

        data1 = self.sources[0]
        data2 = self.sources[1]

        first_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(data1)
        concat_data = DataSource(pd.concat([data1.get_data(), data2.get_data()]), 1)
        second_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(
            concat_data)

        out_sources = [data1.copy(), data1.copy(), data2.copy()]
        out_integrations = [first_integration, second_integration]

        return out_sources, out_integrations

    def get_scenario2(self):
        """
        This method creates the second scenario which consists in two sub-scenarios, where the integration is performed
        correctly and no integration is applied.
        :return: (list of sources, list of correctly integrated datasets, list of non integrated datasets)
        """

        sources, perfect_integrations = self.get_scenario2_match()
        _, wrong_integrations = self.get_scenario2_no_match()

        return sources, perfect_integrations, wrong_integrations

    def get_scenario3_match(self):
        """
        This method creates the scenario 'scenario3_match' which consists into 3 different data sources and two
        integrated datasets. These integrated datasets are obtained by incrementally integrating the three data sources.
        :return: (a list of data sources, a list of integrated datasets)
        """

        integration1 = self.integrations[0]
        integration2 = self.integrations[1]
        first_integration = DataFusionComponent(integration1).select_random_records(self.random_seed)
        second_integration = DataFusionComponent(integration2).select_random_records(self.random_seed)

        return [source.copy() for source in self.sources], [first_integration, second_integration]

    def get_scenario3_no_match(self):
        """
        This method creates the scenario 'scenario3_no_match' which consists into 3 different data sources and two
        integrated datasets. These integrated datasets are obtained by incrementally concatenating the three data
        sources (i.e., no integration is applied).
        :return: (a list of data sources, a list of integrated datasets)
        """

        data1 = self.sources[0]
        data2 = self.sources[1]
        data3 = self.sources[2]

        two_sources_concat_data = DataSource(pd.concat([data1.get_data(), data2.get_data()]), 0)
        first_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(
            two_sources_concat_data)

        three_sources_concat_data = DataSource(pd.concat([data1.get_data(), data2.get_data(), data3.get_data()]), 1)
        second_integration = SimpleIntegrationScenariosGeneration._create_integrated_dataset_from_data_source(
            three_sources_concat_data)

        out_sources = [data1.copy(), data2.copy(), data3.copy()]
        out_integrations = [first_integration, second_integration]

        return out_sources, out_integrations

    def get_scenario3(self):
        """
        This method creates the third scenario which consists in two sub-scenarios, where the integration is performed
        correctly and no integration is applied.
        :return: (list of sources, list of correctly integrated datasets, list of non integrated datasets)
        """

        sources, perfect_integrations = self.get_scenario3_match()
        _, wrong_integrations = self.get_scenario3_no_match()

        return sources, perfect_integrations, wrong_integrations


class ScenarioGenerator(object):
    """
    This class manages the creation of data integration scenarios.
    """

    def __init__(self, integration, random_seed):
        """
        This method stores the single user-provided integrated dataset and exploits its content to create multiple data
        integration scenarios.
        :param integration: IntegrationDataset object from which to generate the data integration scenarios
        :param random_seed: seed for random choices
        """

        if not isinstance(integration, IntegratedDataset):
            raise TypeError("Wrong data type for parameter integration. Only IntegratedDataset data type is allowed.")

        if not isinstance(random_seed, int):
            raise TypeError("Wrong data type for parameter random_seed. Only integrer data type is allowed.")

        self.integration = integration
        self.random_seed = random_seed

        self.integration_entity_id = integration.get_entity_label_col()
        self.integration_source_id = integration.get_source_id_col()
        self.integration_id = integration.get_id_col()

    def _create_integrated_dataset_from_data_source(self, data_source):
        """
        This method transforms a data source in an integrated dataset.
        :param data_source: the data source to be transformed in integrated dataset
        :return: the IntegratedDataset object obtained from the content of the input data source
        """

        if not isinstance(data_source, DataSource):
            raise TypeError("Wrong data type for parameter data_source. Only DataSource data type is allowed.")

        integration_data = data_source.get_data().copy()
        entity_ids = range(len(integration_data))
        integration_data[self.integration_entity_id] = entity_ids
        integration_data[self.integration_source_id] = data_source.get_source_id()
        integration_data.reset_index(inplace=True)
        integrated_dataset = IntegratedDataset(integration_data, self.integration_id, self.integration_source_id,
                                               self.integration_entity_id)

        return integrated_dataset

    def get_scenario_duplicates_impact(self):
        """
        This method generates a data integration scenario which aims to evaluate the impact of duplications on the
        metric used for profiling the data integration task. Two sub-scenarios are generated.
        Sub-scenario A uses the same data (S1) both for representing data sources and integration datasets.
        The sub-scenario B modifies the sub-scenario A by substituting the last integrated dataset with the
        concatenation of the only data considered (S1|S1).
        # DATA SOURCES   A)  S1     S1      S1     B)  S1      S1      S1
        # INTEGRATIONS          S1      S1                S1     S1|S1
        :return: (data sources and integrated datasets for sub-scenario A, data sources and integrated datasets for
                 sub-scenario B)
        """

        # extract from the integrated dataset some data sources
        sources, integrations = self.integration.generate_datasets_for_data_integration(data_type="all",
                                                                                        same_size=False)
        s1 = sources[0]
        s1_data = s1.get_data()
        s1_source_id = s1.get_source_id()

        data_sources_AB = [s1.copy(), s1.copy(), s1.copy()]

        s1_integration = self._create_integrated_dataset_from_data_source(s1)
        s1_concat_source = DataSource(pd.concat([s1_data, s1_data]), s1_source_id)
        s1_concat_integration = self._create_integrated_dataset_from_data_source(s1_concat_source)

        integrations_A = [s1_integration.copy(), s1_integration.copy()]
        integrations_B = [s1_integration.copy(), s1_concat_integration.copy()]

        return data_sources_AB, integrations_A, data_sources_AB, integrations_B

    def get_scenario_integration_by_source_impact(self):
        """
        This method generates a data integration scenario which aims to evaluate the impact of each data source in the
        hypothesis that the data fusion logic, in case of conflicts, selects always the record from the same data
        source. Two sub-scenarios are generated.
        Sub-scenario A integrates two same-sized data sources (S1 and S2) and the last integrated dataset is built by
        preferring, in case of conflicts, the records from the first data source.
        The sub-scenario B modifies the sub-scenario A by substituting the last integrated dataset with S2 (i.e., the
        data fusion logic prefers in case of conflicts the records of the second data source).
        In order to remove noisy data, only matched records from S1 and S2 are considered.
        # DATA SOURCES   A)  S1     S1      S2     B)  S1      S1      S2
        # INTEGRATIONS          S1      S1                S1       S2
        :return: (data sources and integrated datasets for sub-scenario A, data sources and integrated datasets for
                 sub-scenario B)
        """

        # extract from the integrated datasets some data with equal size
        data_sources, integrations = self.integration.generate_datasets_for_data_integration(data_type="match",
                                                                                             same_size=True)
        s1 = data_sources[0]
        s2 = data_sources[1]

        data_sources_AB = [s1.copy(), s1.copy(), s2.copy()]

        s1_integration = self._create_integrated_dataset_from_data_source(s1)
        s2_integration = self._create_integrated_dataset_from_data_source(s2)

        integrations_A = [s1_integration.copy(), s1_integration.copy()]
        integrations_B = [s1_integration.copy(), s2_integration.copy()]

        return data_sources_AB, integrations_A, data_sources_AB, integrations_B

    def get_scenario_size_impact(self):
        """
        This method generates a data integration scenario which aims to evaluate the impact of the size of the data to
        be integrated. Two sub-scenarios are generated.
        Sub-scenario A integrates two data sources (S1 and S2) where S2 is greater than S1. In the sub-scenario B S2 is
        smaller than S1. In order to remove noisy data, only matched records from S1 and S2 are considered.
        # DATA SOURCES   A)  S1     S1      S2     B)  S1      S1      S2
        # INTEGRATIONS          S1      S2                S1       S2
        :return: (data sources and integrated datasets for sub-scenario A, data sources and integrated datasets for
                 sub-scenario B)
        """

        # extract from the integrated datasets some data
        match_data_sources, match_integrations = self.integration.generate_datasets_for_data_integration(
            data_type="match", same_size=False)
        s1 = match_data_sources[0]
        s2 = match_data_sources[1]

        # sub-scenario A: S2 bigger than S1
        # take only 1/3 of data of S1
        s1_data = s1.get_data()
        s1_small_data = s1_data.iloc[:int(len(s1_data) / 3), :]
        s1_small = DataSource(s1_small_data, s1.get_source_id())

        s1_integration_small = self._create_integrated_dataset_from_data_source(s1_small)
        s2_integration = self._create_integrated_dataset_from_data_source(s2)

        data_sources_A = [s1_small.copy(), s1_small.copy(), s2.copy()]
        integrations_A = [s1_integration_small.copy(), s2_integration.copy()]

        # sub-scenario B: S2 smaller than S1
        # take only 1/3 of data of S2
        s2_data = s2.get_data()
        s2_small_data = s2_data.iloc[:int(len(s1_data) / 3), :]
        s2_small = DataSource(s2_small_data, s2.get_source_id())

        s2_integration_small = self._create_integrated_dataset_from_data_source(s2_small)
        s1_integration = self._create_integrated_dataset_from_data_source(s1)

        data_sources_B = [s1.copy(), s1.copy(), s2_small.copy()]
        integrations_B = [s1_integration.copy(), s2_integration_small.copy()]

        return data_sources_A, integrations_A, data_sources_B, integrations_B

    def get_scenario_information_type_impact(self):
        """
        This method generates a data integration scenario which aims to evaluate the impact of the type of information
        integrated in a data integration task. Two types of information are considered: new and old information.
        Two sub-scenarios are generated.
        Sub-scenario A integrates two data sources (S1 and S2) where S2 contains only information that match with S1.
        In the sub-scenario B S2 contains information that doesn't match with S1 (i.e., it is a new kind of
        information). The last integrated dataset for this sub-scenario is obtained by concatenating S1 with S2.
        In order to remove noisy data, each sub-scenario uses same-sized datasets.
        # DATA SOURCES   A)  S1     S1      S2     B)  S1      S1      S2
        # INTEGRATIONS          S1      S2                S1      S1|S2
        :return: (data sources and integrated datasets for sub-scenario A, data sources and integrated datasets for
                 sub-scenario B)
        """
        match_data_sources, match_integrations = self.integration.generate_datasets_for_data_integration(
            data_type="match",
            same_size=True)

        non_match_data_sources, non_match_integrations = self.integration.generate_datasets_for_data_integration(
            data_type="non_match",
            same_size=True)

        # sub-scenario A
        s1_match = match_data_sources[0]
        s2_match = match_data_sources[1]
        s1_match_integration = self._create_integrated_dataset_from_data_source(s1_match)
        s2_match_integration = self._create_integrated_dataset_from_data_source(s2_match)

        data_sources_A = [s1_match.copy(), s1_match.copy(), s2_match.copy()]
        integrations_A = [s1_match_integration.copy(), s2_match_integration.copy()]

        # sub-scenario B
        s1_non_match = non_match_data_sources[0]
        s2_non_match = non_match_data_sources[1]
        s1_non_match_integration = self._create_integrated_dataset_from_data_source(s1_non_match)
        s1_s2_concat_non_match = non_match_integrations[0]

        data_sources_B = [s1_non_match.copy(), s1_non_match.copy(), s2_non_match.copy()]
        integrations_B = [s1_non_match_integration.copy(), s1_s2_concat_non_match.copy()]

        return data_sources_A, integrations_A, data_sources_B, integrations_B

    def get_scenario_integration_result_impact(self):
        """
        This method generates a data integration scenario which aims to evaluate the impact of a correctly integrated
        dataset over a wrong integrated dataset in a data integration task.
        Two sub-scenarios are generated.
        Sub-scenario A integrates two data sources (S1 and S2) where the last integrated dataset is obtained by
        integrating S1 and S2 (in case of conflict, the data fusion logic selects randomly the record to be inserted in
        the output integration result). In the sub-scenario B the last integrated dataset is obtained by concatenating
        S1 with S2 (i.e., no integration is applied).
        In order to remove noisy data, each sub-scenario uses same-sized datasets and considers only matched records.
        # DATA SOURCES   A)  S1     S1      S2     B)  S1      S1      S2
        # INTEGRATIONS          S1    S1+S2                S1      S1|S2
        :return: (data sources and integrated datasets for sub-scenario A, data sources and integrated datasets for
                 sub-scenario B)
        """

        match_data_sources, match_integrations = self.integration.generate_datasets_for_data_integration(
            data_type="match",
            same_size=True)
        s1 = match_data_sources[0]
        s2 = match_data_sources[1]
        s1_s2_concat_integration = match_integrations[0]
        s1_integration = self._create_integrated_dataset_from_data_source(s1)
        data_sources_AB = [s1.copy(), s1.copy(), s2.copy()]

        # sub-scenario A: random data fusion
        data_fusion_comp = DataFusionComponent(s1_s2_concat_integration)
        s1_s2_integration = data_fusion_comp.select_random_records(self.random_seed)

        integrations_A = [s1_integration.copy(), s1_s2_integration.copy()]

        # sub-scenario B: concatenation
        integrations_B = [s1_integration.copy(), s1_s2_concat_integration.copy()]

        return data_sources_AB, integrations_A, data_sources_AB, integrations_B


# NEW PART
class ScenarioContainer(object):
    """
    This class implements a general scenario container.
    Each scenario is composed of two components:
     - scenario parameters
     - scenario data
    Optionally the container can store some scores computed by some metric over the considered scenario.
    """
    def __init__(self, params: dict, data: list):
        """
        This method initializes the state of the scenario container.

        :param params: dictionary containing the scenario parameters
        :param data: list containing the scenario data
        """
        check_parameter_type(params, 'params', dict, 'dictionary')
        check_parameter_type(data, 'data', list, 'list')

        self.params = params
        self.data = data
        self.scores = {}

    def get_params(self):
        """
        This method returns the scenario parameters.
        """
        return self.params

    def get_data(self):
        """
        This method returns the scenario data.
        """
        return self.data

    def get_scores(self):
        """
        This method returns the scenario scores.
        """
        return self.scores


class DataIntegrationScenarioContainer(ScenarioContainer):
    """
    This class implements a data integration scenario container: a scenario container where the data and the scores
    refer to a data integration task.
    """
    def __init__(self, params: dict, data: list):
        """
        This method initializes the state of the container.

        :param params: dictionary containing the scenario parameters
        :param data: list containing the scenario data
        """
        super().__init__(params, data)

    def save_metric_scores(self, metric_params: dict, scores: list, mean: bool, num_sources: int,
                           extra_results: list = None):
        """
        This method saves some scores computed by some metric over the current scenario.

        :param metric_params: the parameters of the metric which has generated the scores
        :param scores: the scores computed by some metric over the current scenario
        :param mean: boolean flag that indicates whether the provided scores have been aggregated by mean
        :param num_sources: the number of data sources involved in the data integration scenario
        :param extra_results: optional list of extra results
        :return: None
        """

        check_parameter_type(metric_params, 'metric_params', dict, 'dictionary')
        check_parameter_type(scores, 'scores', list, 'list')
        check_parameter_type(mean, 'mean', bool, 'boolean')
        check_parameter_type(extra_results, 'extra_results', list, 'list', optional_param=True)
        check_parameter_type(num_sources, 'num_sources', int, 'integer')

        scenario_data = self.data

        if len(scores) != len(scenario_data):
            raise ValueError("Wrong number of scores provided.")

        if extra_results is not None:
            if len(extra_results) != len(scores):
                raise ValueError("Wrong number of extra results provided.")

        if num_sources <= 0:
            raise ValueError("Wrong value for parameter num_sources. Only positive values are allowed.")

        score_template = self.params
        score_template.update(metric_params)

        distance_range_mapper = interp1d([0, np.sqrt(2)], [0, 1])
        # y_range_mapper = interp1d([1.0 / num_sources, 1], [0, 1])
        y_range_mapper = interp1d([0.4, 1], [0, 1])

        cached_scores = []
        for index, score in enumerate(scores):
            current_scenario_data = scenario_data[index]
            sources = current_scenario_data['sources']
            integrations = current_scenario_data['integrations']
            data_type = current_scenario_data['data_type']

            # consider only the scores between the last integrated dataset and all the data sources
            final_comparison_index = len(score)
            final_comparison_values = score[final_comparison_index]
            final_comparison_sources = final_comparison_values['source']
            final_comparison_integration = final_comparison_values['integration']

            if not mean:

                for k in range(len(final_comparison_sources)):
                    res = score_template.copy()
                    if extra_results is not None:
                        res.update(extra_results[index])
                    res["I"] = len(integrations[0])
                    res["D"] = len(sources[k])
                    res["Ix"] = final_comparison_sources[k]
                    res["Iy"] = final_comparison_integration[k]
                    res["Iy normalized"] = y_range_mapper(res["Iy"])
                    res["score (x,y)"] = np.linalg.norm(np.array([res["Ix"], res["Iy"]]) - np.array((1, 1)))
                    res["score (x,y norm)"] = np.linalg.norm(
                        np.array([res["Ix"], res["Iy normalized"]]) - np.array((1, 1)))
                    res["norm score (x, y)"] = distance_range_mapper(res["score (x,y)"])
                    res["norm score (x, y norm)"] = distance_range_mapper(res["score (x,y norm)"])
                    res["data_type"] = data_type
                    res["datasets"] = "D{}-I".format(k + 1)
                    cached_scores.append(res)

            else:

                mean_res = score_template.copy()
                if extra_results is not None:
                    mean_res.update(extra_results[index])
                mean_res["Ix"] = final_comparison_sources[0]
                mean_res["Iy"] = final_comparison_integration[0]
                mean_res["Iy normalized"] = y_range_mapper(mean_res["Iy"])
                mean_res["score (x,y)"] = np.linalg.norm(np.array([mean_res["Ix"], mean_res["Iy"]]) - np.array((1, 1)))
                mean_res["score (x,y norm)"] = np.linalg.norm(
                    np.array([mean_res["Ix"], mean_res["Iy normalized"]]) - np.array((1, 1)))
                mean_res["norm score (x, y)"] = distance_range_mapper(mean_res["score (x,y)"])
                mean_res["norm score (x, y norm)"] = distance_range_mapper(mean_res["score (x,y norm)"])
                mean_res["data_type"] = data_type
                mean_res["aggregation"] = "MEAN"
                cached_scores.append(mean_res)

        scores_conf = ""
        for k,v in metric_params.items():
            scores_conf += "{}={}_".format(k, v)
        scores_conf = scores_conf[:-1]

        self.scores[scores_conf] = cached_scores
