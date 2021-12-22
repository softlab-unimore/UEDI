import numpy as np
from nltk.translate import bleu_score
# import hashlib

from uedi.utils.general_utilities import check_parameter_type
from uedi.data_integration.data_integration_utilities import check_tokenized_dataset_format
from uedi.nlp.embeddings import GensimEmbeddingManager, SpacyEmbeddingManager
from uedi.nlp.language_models import LanguageModel
from uedi.nlp.similarity_functions import JaccardCalculator


class TwoWayMetricCalculator(object):
    """
    This class defines the interface for a general two-way metric calculator. A "two-way metric" operates over two
    set of objects and computes its score by comparing the first set of objects with respect the second one and
    viceversa.
    """

    @staticmethod
    def check_conf_parameters(params: dict):
        """
        This method checks the provided configuration parameters.

        :param params: the configuration parameters to be checked
        """
        pass

    @staticmethod
    def convert_params_for_export(params: dict):
        """
        This method converts the provided configuration parameters into an export data format.

        :param params: the configuration parameters to be converted for export
        """
        pass

    def calculate_on_two_datasets(self, dataset1: list, dataset2: list, dataset1_id: str, dataset2_id: str,
                                  metric_params: dict, num_sources: int):
        """
        This method applies the two-way metric over two datasets.

        :param dataset1: first dataset for computing the metric score
        :param dataset2: second dataset for computing the metric score
        :param dataset1_id: identifier of the first dataset
        :param dataset2_id: identifier of the second dataset
        :param metric_params: dictionary containing the metric parameters
        :param num_sources: the number of data sources involved in the data integration scenario
        """
        pass

    def calculate_on_multiple_datasets(self, datasets1: list, datasets2: list, datasets1_ids: list, datasets2_ids: list,
                                       metric_params: dict, mean: bool):
        """
        This method applies the two-way metric over multiple datasets.

        :param datasets1: first set of datasets for computing the metric score
        :param datasets2: second set of datasets for computing the metric score
        :param datasets1_ids: identifiers of the first set of datasets
        :param datasets2_ids: identifiers of the second set of datasets
        :param metric_params: dictionary containing the metric parameters
        :param mean: a boolean flag that indicates whether to compute aggregated metric scores
        """
        pass


class RepresentativenessCalculator(TwoWayMetricCalculator):
    """
    This class implements a language-model-based representativeness metric.
    """
    available_modes = ['histogram', 'overlap', 'scaled_histogram', 'difference', 'difference_sum', 'precision',
                       'recall_od', 'recall_bd', 'recall_vocab_bd', 'pr', 'fscore', 'fscore_v2',
                       'jaccard_difference', 'bleu_difference', 'embedding_difference']

    def __init__(self):
        """
        This method initializes the state of the class.
        """
        self.language_models_history = {}
        self.representativeness_history = {}

    @staticmethod
    def check_conf_parameters(params: dict):
        """
        This method checks the provided configuration parameters.

        :param params: the configuration parameters to be checked
        :return: None
        """

        param_names = ['mode', 'ngram', 'collection', 'embed_manager']
        for p in param_names:
            if p not in params:
                raise ValueError("Missing parameters.")

        mode = params['mode']
        ngram = params['ngram']
        collection = params['collection']
        collection_id = params['collection_id']
        embed_manager = params['embed_manager']

        check_parameter_type(mode, 'mode', str, 'string')
        check_parameter_type(ngram, 'ngram', int, 'integer')
        check_parameter_type(collection, 'collection', np.ndarray, 'numpy array', optional_param=True)
        check_parameter_type(collection_id, 'collection_id', str, 'string', optional_param=True)
        check_parameter_type(embed_manager, 'embed_manager', (GensimEmbeddingManager, SpacyEmbeddingManager),
                             'GensimEmbeddingManager/SpacyEmbeddingManager', optional_param=True)

        if mode not in RepresentativenessCalculator.available_modes:
            raise ValueError("Wrong modality selected.")

        if collection is not None:
            for col_dataset in collection:
                check_tokenized_dataset_format(col_dataset, 'collection dataset')

        if mode == 'kl':
            if collection is None:
                raise ValueError("Dataset collection not provided in 'kl' mode.")
        elif mode == 'embedding_difference':
            if embed_manager is None:
                raise ValueError("Embedding manager not provided in 'embedding_difference' mode.")

    @staticmethod
    def convert_params_for_export(params: dict):
        """
        This method converts the provided configuration parameters into an export data format.

        :param params: the configuration parameters to be converted for export
        :return: dictionary containing an exportable version of the parameters
        """

        new_params = {}
        for key in params:
            param = params[key]
            if param is None:
                new_params[key] = 'None'
            else:
                if key == 'embed_manager':
                    param = param.model_to_string()
                new_params[key] = param

        return new_params

    def calculate_on_two_datasets(self, dataset1: np.ndarray, dataset2: np.ndarray, dataset1_id: str, dataset2_id: str,
                                  metric_params: dict, num_sources: int):
        """
        This method computes the language-model-based representativeness between two datasets.

        :param dataset1: first dataset for computing the representativeness score
        :param dataset2: second dataset for computing the representativeness score
        :param dataset1_id: first dataset identifier
        :param dataset2_id: second dataset identifier
        :param metric_params: representativeness configuration parameters
            - mode (str): representativeness modality to use
            - ngram (int, default=1): type of ngram for the language model creation
            - collection (list, default=None): optional list of datasets for applying a KL smoothing technique
            - embed_manager (GensimEmbeddingManager/SpacyEmbeddingManager, default=None): embedding manager for
                embedding-based representativeness modes
        :param num_sources: number of total data sources involved in the considered data integration scenario
        :return: representativeness score expressed in both directions (dataset1 -> dataset2 and dataset2 -> dataset1)
        """

        check_tokenized_dataset_format(dataset1, 'dataset1')
        check_tokenized_dataset_format(dataset2, 'dataset2')
        check_parameter_type(dataset1_id, 'dataset1_id', str, 'string')
        check_parameter_type(dataset2_id, 'dataset2_id', str, 'string')
        check_parameter_type(metric_params, 'metric_params', dict, 'dictionary')
        check_parameter_type(num_sources, 'num_sources', int, 'integer')
        RepresentativenessCalculator.check_conf_parameters(metric_params)

        if num_sources <= 0:
            raise ValueError("Wrong value for parameter num_sources. Only positive values are allowed.")

        mode = metric_params['mode']
        ngram = metric_params['ngram']
        collection = metric_params['collection']
        collection_id = metric_params['collection_id']
        embed_manager = metric_params['embed_manager']

        # create dataset1 language model

        # FIXME: try to automatically create a dataset1 identifier
        # dataset1_id = hashlib.sha1(np.array(dataset1).view(np.uint8)).hexdigest()
        # dataset1_id = hashlib.md5(np.array_str(dataset1).encode('utf-8')).hexdigest()

        if dataset1_id in self.language_models_history:
            lm1 = self.language_models_history[dataset1_id]
        else:
            lm1 = LanguageModel(n=ngram, mtype='mle')
            lm1.fit(dataset1)
            self.language_models_history[dataset1_id] = lm1

        # create dataset2 language model

        # FIXME: try to automatically create a dataset2 identifier
        # dataset2_id = hashlib.sha1(np.array(dataset2).view(np.uint8)).hexdigest()
        # dataset2_id = hashlib.md5(np.array_str(dataset1).encode('utf-8')).hexdigest()

        if dataset2_id in self.language_models_history:
            lm2 = self.language_models_history[dataset2_id]
        else:
            lm2 = LanguageModel(n=ngram, mtype='mle')
            lm2.fit(dataset2)
            self.language_models_history[dataset2_id] = lm2

        if collection is not None:
            # compute collection language model

            # FIXME: try to automatically create a collection identifier
            # collection_id = np.sum([hashlib.md5(np.array_str(d).encode('utf-8')).hexdigest() for d in collection])

            if collection_id in self.language_models_history:
                lm_collection = self.language_models_history[collection_id]
            else:
                docs_collection = dataset1
                for dataset in collection:
                    docs_collection = np.concatenate([docs_collection, dataset])
                # create integrated dataset language model
                lm_collection = LanguageModel(n=ngram, mtype='mle')
                lm_collection.fit(docs_collection)
                self.language_models_history[collection_id] = lm_collection

        export_metric_params = RepresentativenessCalculator.convert_params_for_export(metric_params)
        exp_mode = export_metric_params['mode']
        exp_ngram = export_metric_params['ngram']
        exp_emb = export_metric_params['embed_manager']
        pair_dataset_id = "{}_{}_{}_{}_{}".format(dataset1_id, dataset2_id, exp_mode, exp_ngram, exp_emb)
        if mode == 'kl':
            pair_dataset_id = "{}_{}".format(pair_dataset_id, collection_id)

        if pair_dataset_id in self.representativeness_history:

            print("Retrieved cached results ({}).".format(pair_dataset_id))
            ij, ji = self.representativeness_history[pair_dataset_id]
            print("S->I: {}".format(ji))
            print("I->S: {}".format(ij))

            return ij, ji

        print("Computing new results ({})...".format(pair_dataset_id))

        if mode == 'kl':
            ij = lm1.kl_smoothed(lm2, lm_collection=lm_collection, lambd=0.8)
            ji = lm2.kl_smoothed(lm1, lm_collection=lm_collection, lambd=0.8)

        elif mode == 'histogram':
            ij = lm1.histogram_intersection(lm2)
            ji = lm2.histogram_intersection(lm1)

        elif mode == 'overlap':
            ij = lm1.histogram_overlap(lm2)
            ji = lm2.histogram_overlap(lm1)

        elif mode == 'scaled_histogram':
            ij = lm1.scaled_histogram_intersection(lm2)
            ji = lm2.histogram_intersection(lm1)

        elif mode == 'difference':
            ji, ij = lm2.histogram_difference(lm1, num_sources=num_sources)

        elif mode == 'difference_sum':
            ji, ij = lm2.histogram_difference_sum(lm1)

        elif mode == 'precision':
            ji, ij = lm2.precision(lm1)

        elif mode == 'recall_od':
            ji = lm2.recall(lm1)
            ij = ji
            # ij = lm_integration.recall(lm_source)

        elif mode == 'recall_bd':
            ji = lm2.recall(lm1)
            # ij = ji
            ij = lm1.recall(lm2)

        elif mode == 'recall_vocab_bd':
            ji = lm2.recall_vocab(lm1)
            # ij = ji
            ij = lm1.recall_vocab(lm2)

        elif mode == 'pr':
            _, ij = lm2.precision(lm1)
            ji = lm2.recall(lm1)

        elif mode == 'fscore':
            rji = lm2.recall(lm1)
            # rij = rji
            rij = lm1.recall(lm2)
            pji, pij = lm2.precision(lm1)

            ij = 2 * pij * rij / (pij + rij)
            ji = 2 * pji * rji / (pji + rji)

        elif mode == 'fscore_v2':
            rji = lm2.recall(lm1)
            rij = rji
            # rij = lm1.recall(lm2)
            pji, pij = lm2.precision(lm1)

            ij = 2 * pij * rij / (pij + rij)
            ji = 2 * pji * rji / (pji + rji)

        elif mode == 'jaccard_difference':
            ji = np.mean(JaccardCalculator.text_jaccard_similarity(dataset2, dataset1))

            _, ij = lm2.histogram_difference(lm1, num_sources=num_sources)

        elif mode == 'bleu_difference':
            ji = bleu_score.corpus_bleu([dataset1 for _ in range(len(dataset2))], dataset2)
            _, ij = lm2.histogram_difference(lm1, num_sources=num_sources)

        elif mode == 'embedding_difference':
            ji = embed_manager.compare_embedded_datasets(dataset2, dataset1)
            _, ij = lm2.histogram_difference(lm1, num_sources=num_sources)

        else:
            raise ValueError("Wrong modality.")

        print("S->I: {}".format(ji))
        print("I->S: {}".format(ij))
        self.representativeness_history[pair_dataset_id] = (ij, ji)

        return ij, ji

    def calculate_on_multiple_datasets(self, datasets1: list, datasets2: list, datasets1_ids: list, datasets2_ids: list,
                                       metric_params: dict, mean: bool = False):
        """
        This method computes the language-model-based representativeness between two datasets.

        :param datasets1: the list of data sources involved in the data integration task
        :param datasets2: the list of integrated datasets involved in the data integration task
        :param datasets1_ids: the list of data sources identifiers
        :param datasets2_ids: the list of integration datasets identifiers
        :param metric_params: representativeness configuration parameters
            - mode (str): representativeness modality to use
            - ngram (int, default=1): type of ngram for the language model creation
            - embed_manager (GensimEmbeddingManager/SpacyEmbeddingManager, default=None): embedding manager for
                embedding-based representativeness modes
        :param mean: a boolean flag that indicates whether to compute aggregated metric scores
        :return: set of representativeness scores computed between each pair (data source, integrated dataset)
        """

        sources = datasets1
        integrations = datasets2
        sources_ids = datasets1_ids
        integrations_ids = datasets2_ids

        check_parameter_type(sources, 'sources', list, 'list')
        check_parameter_type(integrations, 'integrations', list, 'list')
        check_parameter_type(sources_ids, 'sources_ids', list, 'list')
        for source_id in sources_ids:
            check_parameter_type(source_id, 'sources_ids item', str, 'string')
        check_parameter_type(integrations_ids, 'integrations_ids', list, 'list')
        for integration_id in integrations_ids:
            check_parameter_type(integration_id, 'integrations_ids item', str, 'string')
        check_parameter_type(metric_params, 'metric_params', dict, 'dictionary')
        check_parameter_type(mean, 'mean', bool, 'boolean')

        for s_data in sources:
            check_tokenized_dataset_format(s_data, "source dataset")

        for i_data in integrations:
            check_tokenized_dataset_format(i_data, "integration dataset")

        parameters = ['mode', 'ngram', 'embed_manager']
        for p in parameters:
            if p not in metric_params:
                raise ValueError("Missing parameters.")

        mode = metric_params['mode']

        res_scores = {}
        for i, docs_integration in enumerate(integrations, 1):
            print("INTEGRATION {}".format(i))
            integration_id = integrations_ids[i-1]

            # compute collection
            collection = None
            collection_id = None
            if mode == 'kl':
                collection = docs_integration
                collection_id = "{}".format(integration_id)
                for j in range(i):
                    collection = np.concatenate([collection, sources[j]])
                    collection_id += "_{}".format(sources_ids[j])

            # iterate over the past examples
            res_scores[i] = {
                'source': [],
                'integration': [],
            }
            for j in range(i):
                print("SOURCE {}".format(j))
                docs_source = sources[j]
                source_id = sources_ids[j]

                metric_params.update({'collection': collection, 'collection_id': collection_id})

                ij, ji = self.calculate_on_two_datasets(docs_integration, docs_source, integration_id, source_id,
                                                        metric_params, len(sources))

                res_scores[i]['source'].append(ji)
                res_scores[i]['integration'].append(ij)

        if mean:
            average_res = {}
            for k, val in res_scores.items():
                average_res[k] = {}
                average_res[k]['integration'] = [np.mean(val['integration'])]
                average_res[k]['source'] = [np.mean(val['source'])]
            res_scores = average_res

        return res_scores
