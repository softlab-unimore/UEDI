import os
import pandas as pd
from uedi.data_integration.data_preparation import DataPreparationComponent
from uedi.evaluation.scenarios import ScenarioGenerator
from uedi.nlp.language_models import LanguageModel
from uedi.nlp.pre_processing import TextPreparation
import collections


SCENARIOS = ["duplicate", "source", "size", "info_type", "integration_result"]



def get_distribution(docs):

    # create language model
    lm = LanguageModel(n=1, model='mle')
    lm.fit(docs)

    vocabs, scores = lm.get_distribution()

    return pd.Series(scores, index=vocabs)

def save_scenario_file(source_records, integration_records, out_dir, name='prova.csv'):
    s1 = get_distribution(source_records)
    s2 = get_distribution(integration_records)
    ds = pd.concat([s1, s2], axis=1)
    ds.fillna(0, inplace=True)

    ds.columns = ['source', 'integrations']
    index = [x.replace(',', '') for x in ds.index]
    ds.index = index

    out_file_name = os.path.join(out_dir, name)
    ds.to_csv(out_file_name)

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

    if scenario not in SCENARIOS:
        raise ValueError("Wrong data value for parameter scenario. Only values {} are allowed.".format(SCENARIOS))

    random_seed = 24
    _, integrations = data_prep_comp.get_all_data()
    integration = integrations[0]

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


def save_example_scenario(example, attrs, scenarios=SCENARIOS):
    """
    This function generates and save into the disk multiple scenarios for testing the same data example.

    :param example: name of the data example
    :param attrs: attributes to consider for the data
    :param scenarios: name of the scenario to be tested
    :return: None
    """

    if not isinstance(example, str):
        raise TypeError("Wrong data type for parameter example. Only string data type is allowed.")

    if not isinstance(scenarios, collections.Iterable):
        raise TypeError("Wrong data type for parameter scenarios. Only iterable data type is allowed.")

    if not isinstance(attrs, collections.Iterable):
        raise TypeError("Wrong data type for parameter attrs. Only iterable data type is allowed.")

    examples = ["restaurant"]

    if example not in examples:
        raise ValueError("Wrong data value for parameter example. Only values {} are allowed.".format(examples))

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError("Wrong data value for scenarios elements. Only values {} are allowed.".format(SCENARIOS))

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    experiment_dir = os.path.join(root_dir, "data_for_experiments")
    example_dir = os.path.join(data_dir, example)

    data_prep_comp = None
    if example == "restaurant":

        # source 1
        file1 = os.path.join(example_dir, "original", "zomato.csv")
        source1_id = 0

        # source 2
        file2 = os.path.join(example_dir, "original", "yelp.csv")
        source2_id = 1

        # integrated data
        file3 = os.path.join(example_dir, "original", "labeled_data.csv")

        data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3)

        # # describe integrated data
        # _, integrations = data_prep_comp.get_all_data()
        # for integration in integrations:
        #     integration.describe()

    for scenario_id, scenario in enumerate(scenarios, 1):
        sources_A, integrations_A, sources_B, integrations_B = generate_scenarios(data_prep_comp, scenario)

        assert len(integrations_A) == 2
        assert len(integrations_B) == 2

        integration = integrations_A[-1]
        integration_id = len(integrations_A)

        # clean integration data
        integration_data = integration.get_data()[attrs]
        integration_attrs = integration_data.columns.values
        clean_integration_data = TextPreparation.convert_dataframe_to_text(integration_data, integration_attrs)

        for source_id, source in enumerate(sources_A):

            # clean source data
            source_data = source.get_data()[attrs]
            source_attrs = source_data.columns.values
            clean_source_data = TextPreparation.convert_dataframe_to_text(source_data, source_attrs)

            save_scenario_file(clean_source_data, clean_integration_data, experiment_dir,
                               "scenario{}_caseA_s{}_i{}.csv".format(scenario_id, source_id, integration_id))

        integration = integrations_B[-1]
        integration_id = len(integrations_B)

        # clean integration data
        integration_data = integration.get_data()[attrs]
        integration_attrs = integration_data.columns.values
        clean_integration_data = TextPreparation.convert_dataframe_to_text(integration_data,
                                                                           integration_attrs)

        for source_id, source in enumerate(sources_B):
            # clean source data
            source_data = source.get_data()[attrs]
            source_attrs = source_data.columns.values
            clean_source_data = TextPreparation.convert_dataframe_to_text(source_data, source_attrs)

            save_scenario_file(clean_source_data, clean_integration_data, experiment_dir,
                               "scenario{}_caseB_s{}_i{}.csv".format(scenario_id, source_id, integration_id))


if __name__ == '__main__':
    attrs = ["NAME", "ADDRESS"]
    save_example_scenario("restaurant", attrs)