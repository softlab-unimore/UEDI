import pandas as pd
import os

from uedi.evaluation import DataIntegrationScenarioContainer
from uedi.nlp import SpacyEmbeddingManager, GensimEmbeddingManager
from uedi.metrics.metrics import RepresentativenessCalculator
from uedi.plot import SimplePlotManager, MultiEntityTypesPlotManager
from uedi.data_integration.data_integration_utilities import dataset_tokenizer


if __name__ == '__main__':

    # STEP 1: DATASET & INPUT PARAMETERS SELECTION

    input_dir = os.path.join(os.path.abspath('.'), 'data', 'running_example')
    input_sources_dir = os.path.join(input_dir, 'sources')
    input_integrations_dir = os.path.join(input_dir, 'integrations')
    out_dir = os.path.join(os.path.abspath('.'), 'data', 'output', 'results', 'running_example')
    columns = ['authors', 'title', 'venue', 'year']

    # data sources
    sources = []
    sources_ids = []
    for source_file_name in os.listdir(input_sources_dir):
        source_file = os.path.join(input_sources_dir, source_file_name)

        if os.path.isfile(source_file) and source_file_name.endswith('.csv'):

            source_id = source_file_name.replace(".csv", "")
            source_data = pd.read_csv(source_file, index_col=0)
            source = dataset_tokenizer(source_data, columns=columns)

            sources.append(source)
            sources_ids.append(source_id)

    # integrated datasets
    scenario_data = []
    for integration_file_name in os.listdir(input_integrations_dir):
        integration_file = os.path.join(input_integrations_dir, integration_file_name)

        if os.path.isfile(integration_file) and integration_file_name.endswith('.csv'):
            integration_id = integration_file_name.replace(".csv", "")
            integration_data = pd.read_csv(integration_file, index_col=0)
            integration = dataset_tokenizer(integration_data, columns=columns)
            data_type = 'perfect'
            if integration_id == 'im':
                data_type = 'match'
            elif integration_id == 'ic':
                data_type = 'concat'

            sub_scenario_data = {
                'sources': sources,
                'integrations': [integration, integration],
                'data_type': data_type,
                'sources_ids': sources_ids,
                'integrations_ids': [integration_id, integration_id]
            }
            scenario_data.append(sub_scenario_data)

    # create the data integration scenario
    scenario_params = {'scenario': 'demo'} # add HERE some info about the considered scenario (e.g., datasets sizes)
                                           # this data will be saved in the file of the results
    scenario = DataIntegrationScenarioContainer(scenario_params, scenario_data)

    # STEP 2: REPRESENTATIVENESS MEASURE CONFIGURATION

    # (optional) choose the embedding manager
    # Gensim embedding manager
    # fasttext_model = 'fasttext-wiki-news-subwords-300'
    # fasttext_manager = GensimEmbeddingManager(fasttext_model)
    #
    # word2vec_model = 'word2vec-google-news-300'
    # word2vec_manager = GensimEmbeddingManager(word2vec_model)
    #
    # glove_model = 'glove-wiki-gigaword-300'
    # glove_manager = GensimEmbeddingManager(glove_model)
    #
    # # Spacy embedding manger
    # spacy_model = 'en_core_web_lg'
    # spacy_manager = SpacyEmbeddingManager(spacy_model)

    metric_params = [
        {
            'mode': 'difference',
            'ngram': 1,
            'embed_manager': None
        },
        # {
        #     'mode': 'jaccard_difference',
        #     'ngram': 1,
        #     'embed_manager': None
        # },
        # {
        #     'mode': 'bleu_difference',
        #     'ngram': 1,
        #     'embed_manager': None
        # },
        # {
        #     'mode': 'embedding_difference',
        #     'ngram': 1,
        #     'embed_manager': fasttext_manager
        # },
        # {
        #     'mode': 'embedding_difference',
        #     'ngram': 1,
        #     'embed_manager': word2vec_manager
        # },
        # {
        #     'mode': 'embedding_difference',
        #     'ngram': 1,
        #     'embed_manager': glove_manager
        # },
        # {
        #     'mode': 'embedding_difference',
        #     'ngram': 1,
        #     'embed_manager': spacy_manager
        # }
    ]

    mean = False  # report results aggregated by source

    # STEP 3: REPRESENTATIVENESS COMPUTATION

    metric_calculator = RepresentativenessCalculator()
    scenario_data = scenario.get_data()

    # loop over alternative representativeness measures
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

            # calculate the representativeness
            scenario_score = metric_calculator.calculate_on_multiple_datasets(sources, integrations, sources_ids,
                                                                              integrations_ids, metric_param, mean)
            scenario_scores.append(scenario_score)

        # save the result
        export_metric_param = RepresentativenessCalculator.convert_params_for_export(metric_param)
        scenario.save_metric_scores(export_metric_param, scenario_scores, mean, len(sources))

    # combine the results obtained from multiple representativeness measures
    results = []
    all_scores = scenario.get_scores()
    for score_key in all_scores:
        results += all_scores[score_key]
    results = pd.DataFrame(results)

    # STEP 4: SAVE RESULTS

    # save the results into file
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # STEP 5: PLOT RESULTS

    # plt_manager = SimplePlotManager(results, out_dir)
    if mean:
        multi_subplots = False
    else:
        multi_subplots = True
    plt_manager = MultiEntityTypesPlotManager(results, multi_subplots=multi_subplots, plot_dir=out_dir)
    plt_manager.plot()
    plt_manager.save_plots("demo")
