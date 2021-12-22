from uedi.nlp import SpacyEmbeddingManager, GensimEmbeddingManager
from uedi.data_repos import DataManager
from uedi.data_repos.dm_utilities import DeepMatcherRepoManager
from uedi.evaluation import MultiMatchPercentagesEvaluator, MultiEntityTypesEvaluator, \
    MinimalityTotalityEvaluator, \
    RepresentativenessVarianceEvaluator, TimePerformanceEvaluator, IntegratedEntitiesVariationEvaluator, \
    MultiSourceIntegrationEvaluator, RepresentativenessBiasEvaluator
from uedi.data_repos.cora_utilities import get_multi_data_sources
from uedi.metrics.metrics import RepresentativenessCalculator
from uedi.plot import MultiEntityTypesPlotManager, MultiMatchPercentagesPlotManager, SimplePlotManager

import os


def run_experiment(repository_id: str, dataset_id: str, experiment_type: str, conf_params: list, mean: bool):
    """
    This function runs an experiment.

    :param repository_id: the name of the repository where experiment data resides
    :param dataset_id: identifier of the data for the experiment
    :param experiment_type: the type of experiment to run
    :param conf_params: list of experiment configuration parameters
    :param mean: boolean flag that indicates whether to compute aggregated results
    :return None
    """

    features = None
    th = None

    if repository_id == 'deep-matcher':

        if dataset_id == 'Structured_DBLP-GoogleScholar' or dataset_id == 'Dirty_DBLP-GoogleScholar':
            features = ['title', 'authors']
            th = 0.5
        elif dataset_id == 'Structured_DBLP-ACM' or dataset_id == 'Dirty_DBLP-ACM':
            features = ['title', 'authors']
            th = 0.5
        elif dataset_id == 'Structured_Amazon-Google':
            features = ['title', 'manufacturer']
            th = 0.5
        elif dataset_id == 'Structured_Walmart-Amazon' or dataset_id == 'Dirty_Walmart-Amazon':
            features = ['title', 'category', 'brand']
            th = 0.5
        elif dataset_id == 'Structured_Beer':
            features = ['Beer_Name', 'Brew_Factory_Name', 'Style']
            th = 0.5
        elif dataset_id == 'Structured_iTunes-Amazon' or dataset_id == 'Dirty_iTunes-Amazon':
            features = ['Song_Name', 'Artist_Name', 'Album_Name', 'Genre']
            th = 0.5
        elif dataset_id == 'Structured_Fodors-Zagats':
            features = ['name', 'addr']
            th = 0.5
        elif dataset_id == 'Textual_Abt-Buy':
            features = ['name', 'description']
            th = 0.5

    elif repository_id == 'cora':

        if dataset_id == 'cora':
            features = ['author', 'title', 'venue']
            th = 0.5

    print("\n####")
    print("\tREPO: {}".format(repository_id))
    print("\tDATASET: {}".format(dataset_id))
    print("\tATTRS: {}".format(features))
    print("\tTH: {}".format(th))
    print("####\n")

    # STEP 1: DATASET PRE-PROCESSING

    pre_data_manager = DataManager(repository_id, dataset_id)

    # clean data
    pre_data_manager.remove_duplicates(th, features)

    # get data
    data = None
    if experiment_type in ['multi-entity-types', 'time-performance', 'bias']:
        data = pre_data_manager.get_dataset_file("all", "clean")

    elif experiment_type in ['multi-match-percentages', 'entities-variation', 'entities-variation-significance']:
        data = pre_data_manager.get_dataset_file("all_triplet", "clean")

    elif experiment_type == 'minimality-totality' or experiment_type == 'minimality-totality-significance':
        data1 = pre_data_manager.get_dataset_file("tableA", "original")
        data2 = pre_data_manager.get_dataset_file("all", "clean")

    elif experiment_type == 'multi-source-integration':
        data = pre_data_manager.get_dataset_file("all", "original")
        sources = get_multi_data_sources(data)

    else:
        raise ValueError('Wrong evaluator.')

    # STEP 2: EXPERIMENT TYPE SELECTION

    if experiment_type == 'multi-entity-types':
        evaluator = MultiEntityTypesEvaluator(data, th, columns=features)

    elif experiment_type == 'multi-match-percentages':
        # OPTION 1: set match-non_match ratios manually
        # match_non_match_ratios = [
        #     (1.0, 1.0), (0.2, None), (0.5, None), (0.7, None), (0.9, None),
        #     (None, 0.2), (None, 0.5), (None, 0.7), (None, 0.9),
        #     (0.9, 0.2), (0.7, 0.5), (0.5, 0.7), (0.2, 0.9)
        # ]
        # match_non_match_ratios = [
        #     (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
        #     (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
        #     (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
        # ]
        # match_non_match_ratios = [
        #     (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5),
        #     (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (1.0, 1.0),
        # ]
        match_non_match_ratios = [
            (0, 0), (1, 1),
            (0, 0.2), (0, 0.4), (0, 0.6), (0, 0.8), (0, 1),
            (0.2, 0), (0.4, 0), (0.6, 0), (0.8, 0), (1, 0),
        ]
        evaluator = MultiMatchPercentagesEvaluator(data, match_non_match_ratios=match_non_match_ratios,
                                                   columns=features)

        # # OPTION 2: request to automatically generate match-non_match ratios
        # num_scenarios = 30
        # evaluator = MultiMatchPercentagesEvaluator(data, num_scenarios=num_scenarios, columns=features)

    elif experiment_type == 'minimality-totality':
        source_id = 0
        data_fusion_option = 2  # 0 # 1
        evaluator = MinimalityTotalityEvaluator(data1, data2, source_id, data_fusion_option, columns=features)

    elif experiment_type == 'minimality-totality-significance':
        source_id = 0
        data_fusion_option = 2  # 0 # 1

        params = {'source': data1, 'integration': data2, 'source_id': source_id,
                  'data_fusion_option': data_fusion_option, 'columns': features}

        evaluator = RepresentativenessVarianceEvaluator('minimality-totality', params, num_experiments=10)

    elif experiment_type == 'entities-variation-significance':
        # changed_entities_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        changed_entities_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        data_fusion_option = 2  # 0 # 1

        params = {'dataset': data, 'changed_entity_percentages': changed_entities_ratios,
                  'data_fusion_option': data_fusion_option, 'columns': features}

        evaluator = RepresentativenessVarianceEvaluator('entities-variation', params, num_experiments=10)

    elif experiment_type == 'bias':
        sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        params = {'extended_integrated_dataset': data, 'sample_sizes': sample_sizes, 'columns': features}
        evaluator = RepresentativenessBiasEvaluator('bias', params, num_experiments=10)

    elif experiment_type == 'time-performance':
        evaluator = TimePerformanceEvaluator(extended_integrated_dataset=data, columns=features)

    elif experiment_type == 'entities-variation':
        changed_entities_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # changed_entities_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        data_fusion_option = 2  # 0 # 1
        evaluator = IntegratedEntitiesVariationEvaluator(data, changed_entities_ratios, data_fusion_option,
                                                         columns=features)

    elif experiment_type == 'multi-source-integration':
        num_sources = 10
        evaluator = MultiSourceIntegrationEvaluator(sources, num_sources, columns=features)

    else:
        raise ValueError('Wrong evaluator.')

    # STEP 4: RUN THE EXPERIMENT
    metric_calculator = RepresentativenessCalculator()
    results = evaluator.evaluate_metric(metric_calculator, conf_params, mean=mean)
    evaluator.save_results("{}".format(dataset_id))

    # STEP 5: PLOTTING RESULTS
    if experiment_type == 'multi-entity-types':
        if mean:
            multi_subplots = False
        else:
            multi_subplots = True
        plt_manager = MultiEntityTypesPlotManager(results, multi_subplots=multi_subplots)

    elif experiment_type == 'multi-match-percentages':
        # plt_manager = MultiMatchPercentagesPlotManager(results)
        plt_manager = None

    elif experiment_type in ['minimality-totality', 'minimality-totality-significance', 'time-performance',
                             'entities-variation', 'entities-variation-significance', 'bias']:
        plt_manager = None

    elif experiment_type == 'multi-source-integration':
        out_plot_path = os.path.join(os.path.abspath('.'), 'data', 'output', 'plots', 'multi-source-integration')

        for sc in results['scenario'].unique():
            res = results[results['scenario'] == sc]
            plt_manager = SimplePlotManager(res, out_plot_path)
            plt_manager.plot()
            plt_manager.save_plots("{}_sc={}".format(dataset_id, sc))

        plt_manager = None

    else:
        raise ValueError('Wrong evaluator.')

    if plt_manager:
        plt_manager.plot()
        plt_manager.save_plots("{}".format(dataset_id))


def run_multi_experiments(repository_id: str, experiment_type: str, conf_params: list, mean: bool):
    """
    This function runs an experiment over multiple datasets.

    :param repository_id: the name of the repository where experiment data resides
    :param experiment_type: the type of experiment to run
    :param conf_params: list of experiment configuration parameters
    :param mean: boolean flag that indicates whether to compute aggregated results
    :return: None
    """

    dataset_ids = None
    if repository_id == 'deep-matcher':

        dataset_ids = ['Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM', 'Structured_Amazon-Google',
                       'Structured_Walmart-Amazon', 'Structured_Beer', 'Structured_iTunes-Amazon',
                       'Structured_Fodors-Zagats', 'Textual_Abt-Buy', 'Dirty_iTunes-Amazon', 'Dirty_DBLP-ACM',
                       'Dirty_DBLP-GoogleScholar', 'Dirty_Walmart-Amazon']
        # dataset_ids = ['Structured_DBLP-GoogleScholar', 'Structured_Walmart-Amazon', 'Dirty_DBLP-GoogleScholar',
        #                'Dirty_Walmart-Amazon']
        # dataset_ids = ['Structured_Beer']

    elif repository_id == 'cora':
        dataset_ids = ['cora']

    for dataset_id in dataset_ids:
        run_experiment(repository_id, dataset_id, experiment_type, conf_params, mean)


if __name__ == '__main__':

    repository_id = 'deep-matcher'
    # repository_id = 'cora'

    # STEP 1: EXPERIMENT TYPE SELECTION

    # experiment_type = 'multi-entity-types'
    # experiment_type = 'multi-match-percentages'
    # experiment_type = 'minimality-totality'
    # experiment_type = 'entities-variation'
    # experiment_type = 'minimality-totality-significance'
    # experiment_type = 'entities-variation-significance'
    # experiment_type = 'time-performance'
    # experiment_type = 'multi-source-integration'
    experiment_type = 'bias'

    # STEP 2: EXPERIMENT CONFIGURATION PARAMETERS SELECTION

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
    # #
    # # Spacy embedding manger
    # spacy_model = 'en_core_web_lg'
    # spacy_manager = SpacyEmbeddingManager(spacy_model)

    conf_params = [
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

    mean = True

    # STEP 3: DATASET SELECTION

    # option 1: run the experiment over a single dataset

    # repository_id = 'deep-matcher'
    # dataset_id = 'Structured_DBLP-GoogleScholar'
    # # dataset_id = 'Structured_DBLP-ACM'
    # # dataset_id = 'Structured_Amazon-Google'
    # # dataset_id = 'Structured_Walmart-Amazon'
    # # dataset_id = 'Structured_Beer'
    # # dataset_id = 'Structured_iTunes-Amazon'
    # # dataset_id = 'Structured_Fodors-Zagats'
    # # dataset_id = 'Textual_Abt-Buy'
    # # dataset_id = 'Dirty_iTunes-Amazon'
    # # dataset_id = 'Dirty_DBLP-ACM'
    # # dataset_id = 'Dirty_DBLP-GoogleScholar'
    # # dataset_id = 'Dirty_Walmart-Amazon'
    #
    # run_experiment(repository_id, dataset_id, experiment_type, conf_params, mean)

    # option 2: run the experiment over all the datasets
    run_multi_experiments(repository_id, experiment_type, conf_params, mean)
