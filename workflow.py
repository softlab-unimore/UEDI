import os
from uedi.data_integration.data_preparation import DataPreparationComponent
from uedi.evaluation import AdvancedIntegrationScenariosGeneration, \
    SimpleIntegrationScenariosGeneration, ScenarioGenerator
from uedi.nlp import BM25Calculator
from uedi.nlp.similarity_functions import JaccardCalculator
from uedi.nlp.language_models import LanguageModelProfiler
from uedi.profiling import IntegrationProfiler, CompareIntegrationProfiler
from uedi.data_integration.data_fusion import DataFusionComponent


def generate_simple_scenarios(data_prep_comp, scenario):
    """
    This function generates simple data integration scenarios.
    :param data_prep_comp: the class responsible for the generation of the data integration scenarios
    :param scenario: the identifier of the considered scenario
    :return: list of datasets corresponding to different integration scenarios
    """
    random_seed = 24
    original_sources, original_integrations = data_prep_comp.split_randomly_data(3, random_seed, mixed=False,
                                                                                 debug=False)

    scenarios_generator = SimpleIntegrationScenariosGeneration(original_sources, original_integrations)

    if scenario == 1:
        return scenarios_generator.get_scenario1()
    elif scenario == 2:
        return scenarios_generator.get_scenario2()
    elif scenario == 3:
        return scenarios_generator.get_scenario3()
    else:
        raise ValueError("Wrong value for parameter scenario. Only values in the range [1,2,3] are allowed.")


def generate_advanced_scenarios(data_prep_comp):
    """
    This function generates advanced integration scenarios.
    :param data_prep_comp: the class responsible for the generation of the data integration scenarios
    :return: list of datasets corresponding to different integration scenarios
    """
    random_seed = 24
    sources, integrations = data_prep_comp.split_randomly_data(3, random_seed, mixed=False, debug=False)

    # PERFECT SCENARIO
    # apply random data fusion to the original integrated dataset
    perfect_integrations = []
    for integration in integrations:
        perfect_integration = DataFusionComponent(integration).select_random_records(random_seed)
        perfect_integrations.append(perfect_integration)

    # SCENARIO AFFECTED BY ERRORS
    # insert random noise in the original integrated data
    error_rate = 0.5
    scenarios_generator = AdvancedIntegrationScenariosGeneration(sources, integrations)
    integration_ids_to_modify = [0]
    error_rates = [error_rate]
    _, integrations_with_errors = scenarios_generator.generate_wrong_scenario_by_randomly_flipping(
        integration_ids_to_modify, random_seed, error_rates=error_rates)

    # apply random data fusion to the original integrated dataset
    noisy_integrations = []
    for integration_with_errors in integrations_with_errors:
        noisy_integration = DataFusionComponent(integration_with_errors).select_random_records(random_seed)
        noisy_integrations.append(noisy_integration)

    return sources, perfect_integrations, noisy_integrations


def generate_scenarios(data_prep_comp, scenario):
    """
    This function generates simple data integration scenarios.
    :param data_prep_comp: the class responsible for the generation of the data integration scenarios
    :param scenario: the string identifier of the considered scenario
    :return: list of datasets corresponding to different integration scenarios
    """

    if not isinstance(data_prep_comp, DataPreparationComponent):
        raise TypeError(
            "Wrong data type for parmeter data_prep_comp. Only DataPreparationComponent data type is allowed.")

    if not isinstance(scenario, str):
        raise TypeError("Wrong data type for parmeter scenario. Only string data type is allowed.")

    scenarios = ["duplicate", "source", "size", "info_type", "integration_result"]
    if scenario not in scenarios:
        raise ValueError("Wrong data value for parameter scenario. Only values {} are allowed.".format(scenarios))

    random_seed = 24
    _, integrations = data_prep_comp.get_all_data()
    integration = integrations[0]

    # # FIXME: in a real test scenario consider all the original integrated dataset and not a partition
    # sample_size = 100
    # integration_data = integration.get_data()
    # integration_match_data = integration_data.groupby(integration.entity_label_col).filter(
    #     lambda x: len(x) > 1)
    # matched_entities = integration_match_data[integration.get_entity_label_col()].unique()
    # random.seed(random_seed)
    # sample_matched_entities = random.choices(matched_entities, k=int(sample_size/2))
    # integration_match_data = integration_match_data[integration_match_data[integration.get_entity_label_col()].isin(sample_matched_entities)]
    # integration_non_match_data = integration_data.groupby(integration.entity_label_col).filter(
    #     lambda x: len(x) == 1).sample(sample_size, random_state=random_seed)
    # integration_concat_data = pd.concat([integration_match_data, integration_non_match_data])
    # integration = IntegratedDataset(integration_concat_data, integration.get_id_col(), integration.get_source_id_col(),
    #                                 integration.get_entity_label_col())

    scenario_generator = ScenarioGenerator(integration, random_seed)

    if scenario == "duplicate":
        return scenario_generator.get_scenario_duplicates_impact()
    elif scenario == "source":
        return scenario_generator.get_scenario_integration_by_source_impact()
    elif scenario == "size":
        return scenario_generator.get_scenario_size_impact()
    elif scenario == "info_type":
        return scenario_generator.get_scenario_information_type_impact()
    elif scenario == "integration_result":
        return scenario_generator.get_scenario_integration_result_impact()


def run_scenario(metric, sources, integrations, profiler_integrator, attrs=None):
    """
    This function takes in input N data sources and N - 1 integrated datasets and profiles the data integration task by
    analysing each data integration iteration with a user provided metric.

    :param metric: the metric name to use to profile the data integration task
    :param sources: list of data sources
    :param integrations: list of integrated datasets
    :param profiler_integrator: profiler object to store results of the profilation
    :param attrs: the optional list of attributes to be considered in the data
    :return: None
    """

    iteration = 1
    # loop over data sources
    for i, source in enumerate(sources):

        if i == 0:
            continue

        print("\tIteration#{}".format(i))

        # get the integration dataset
        integration_dataset = integrations[i - 1]
        integration = integration_dataset.get_data()[attrs]

        num_sources_in_iteration = i
        integration_scores = []

        # loop over the history of processed data sources
        for si_index, si in enumerate(sources):

            print("\t\tSource{} - Integration{}".format(si_index, i - 1))

            # profile data source - integrated dataset pair
            si_data = si.get_data()[attrs]

            source_to_integration = None
            integration_to_source = None

            # FIXME: I'm waiting to model the profiling metric because I don't know how to model it in a general way
            if metric == "BM25":
                # topk = num_sources_in_iteration
                topk = 1
                source_to_integration, integration_to_source = BM25Calculator.get_datasets_bm25(si_data, integration,
                                                                                           topk=topk,
                                                                                           debug=True)
            elif metric == "Jaccard":
                source_to_integration, integration_to_source = JaccardCalculator.get_datasets_Jaccard(si_data,
                                                                                                      integration,
                                                                                                      topk=1,
                                                                                                      debug=False)
            elif metric == "KL":
                source_to_integration, integration_to_source = LanguageModelProfiler.get_dataset_language_model_distance(
                    si_data, integration, topk=1, debug=True)

            elif metric == "Hist overlap":
                source_to_integration, integration_to_source = LanguageModelProfiler.get_dataset_language_model_distance(
                    si_data, integration, topk=1, debug=False)

            scores = {
                'source_to_integration': source_to_integration,
                'integration_to_source': integration_to_source,
            }
            integration_scores.append(scores)

            if si == source:
                break

        iteration += 1

        # update the history of the integration scores
        profiler_integrator.add_integration_scores(i, integration_scores)


def plot_profiler_results(profiler):
    """
    This function plots the profiler results in different formats.
    :param profiler: the profiler from which to extract the scores to plot
    :return: None
    """

    # plot aggregated profiler scores
    profiler.plot_average_history_scores()
    # plot aggregated and single profiler scores
    profiler.plot_average_and_single_history_scores()
    # plot histograms by source
    # profiler.plot_histograms_by_sources()
    # plot histograms by integrations
    # profiler.plot_histograms_by_integrations()


def plot_multi_profiler_results(profilers, profilers_labels):
    """
    This function plots multiple profiler results in the same chart in order to compare easily their results.
    :param profilers: list of profilers to compare
    :param profilers_labels: list of profiler names
    :return: None
    """

    cmp_integration_profilers = CompareIntegrationProfiler(profilers, profilers_labels)
    cmp_integration_profilers.plot_average_history_scores(subplots=False)
    cmp_integration_profilers.plot_average_and_single_history_scores(subplots=True)


def main():
    # STEP 1: get data

    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data")
    example = 'restaurant'
    example_dir = os.path.join(data_dir, example)

    # source 1
    file1 = os.path.join(example_dir, "original", "zomato.csv")
    source1_id = 0

    # source 2
    file2 = os.path.join(example_dir, "original", "yelp.csv")
    source2_id = 1

    # integrated data
    file3 = os.path.join(example_dir, "original", "labeled_data.csv")

    # END STEP 1

    # STEP 2: prepare data

    data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3)

    # # describe integrated data
    # _, integrations = data_prep_comp.get_all_data()
    # for integration in integrations:
    #     integration.describe()

    # simple integration scenarios
    # scenario = 3
    # sources, perfect_integrations, wrong_integrations = generate_simple_scenarios(data_prep_comp, scenario)
    # sources_A = sources
    # sources_B = sources
    # integrations_A = perfect_integrations
    # integrations_B = wrong_integrations

    # advanced integration scenarios
    # sources, perfect_integrations, wrong_integrations = generate_advanced_scenarios(data_prep_comp)
    # sources_A = sources
    # sources_B = sources
    # integrations_A = perfect_integrations
    # integrations_B = wrong_integrations

    # integration scenarios
    scenario = "duplicate"
    # scenario = "source"
    # scenario = "size"
    # scenario = "info_type"
    # scenario = "integration_result"
    sources_A, integrations_A, sources_B, integrations_B = generate_scenarios(data_prep_comp, scenario)

    # describe integration datasets
    attrs = ["NAME", "ADDRESS"]
    print("SOURCES A")
    for sa in sources_A:
        print(sa.get_data().shape)
        # print(sa.get_data())

    print("INTEGRATIONS A")
    for ix, ia in enumerate(integrations_A):
        print(ia.get_data().shape)
        if ix == 1:
            print(ia.get_data().loc[:, attrs + ["source"]])

    print("SOURCES B")
    for sb in sources_B:
        print(sb.get_data().shape)
        # print(sb.get_data())

    print("INTEGRATIONS B")
    for ix, ib in enumerate(integrations_B):
        print(ib.get_data().shape)
        if ix == 1:
            print(ib.get_data().loc[:, attrs + ["source"]])

    # END STEP 2

    # STEP 3: profile integration scenarios

    # metric_name = "BM25"
    # metric_name = "Jaccard"
    # metric_name = "KL"
    metric_name = "Hist overlap"

    # run sub-scenario A
    profiler_A = IntegrationProfiler(metric_name)
    print("\nSUB-SCENARIO A")
    run_scenario(metric_name, sources_A, integrations_A, profiler_A, attrs=attrs)

    # run noisy scenario
    profiler_B = IntegrationProfiler(metric_name)
    print("\nSUB-SCENARIO B")
    run_scenario(metric_name, sources_B, integrations_B, profiler_B, attrs=attrs)

    # END STEP 3

    # STEP 4: Plot profiling results

    # plot single profiler results
    # plot_profiler_results(profiler_A)
    # plot_profiler_results(profiler_B)

    # plot profiler result comparison
    profilers = [profiler_A, profiler_B]
    # profilers_labels = ["Integration", "Concatenation"]
    profilers_labels = ["Sub-scenario A", "Sub-scenario B"]
    plot_multi_profiler_results(profilers, profilers_labels)

    # END STEP 4


if __name__ == '__main__':
    main()
