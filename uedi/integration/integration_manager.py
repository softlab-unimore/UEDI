import py_entitymatching as em
import os
import pandas as pd
import networkx as nx
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logging.getLogger("py_entitymatching").setLevel(logging.WARNING)  # disable INFO logger messages

logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent


class DataLoaderComponent(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_dataset(self, dataset):
        self.logger.info(
            "Loading dataset {}".format(dataset))

        # FIXME: ipotizzo che il dataset di input abbia una colonna 'id' con un indice progressivo
        D = em.read_csv_metadata(os.path.join(ROOT_DIR, dataset), key='id')
        self.logger.info("Dataset: {}.".format(D.shape))

        return D

    def load_magellan_datasets(self, dataset1, dataset2, gt_dataset):
        self.logger.info(
            "Loading Magellan datasets A: {}, B: {}, ground truth: {}.".format(dataset1, dataset2, gt_dataset))
        # Get the datasets directory
        datasets_dir = em.get_install_path() + os.sep + 'datasets'

        path_A = datasets_dir + os.sep + dataset1
        path_B = datasets_dir + os.sep + dataset2
        path_labeled_data = datasets_dir + os.sep + gt_dataset

        A = em.read_csv_metadata(path_A, key='id')
        B = em.read_csv_metadata(path_B, key='id')
        self.logger.info("Dataset A: {}.".format(A.shape))
        self.logger.info("Dataset B: {}.".format(B.shape))

        # Load the pre-labeled data
        ground_truth = em.read_csv_metadata(path_labeled_data,
                                            key='_id',
                                            ltable=A, rtable=B,
                                            fk_ltable='ltable_id', fk_rtable='rtable_id')
        self.logger.info("Ground truth: {}.".format(ground_truth.shape))

        self.logger.info("Datasets loaded successfully.".format(ground_truth.shape))

        return A, B, ground_truth


class DataPreprocessingComponent(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_prefix_to_identifier_column(self, data, identifier_column, prefix):

        self.logger.info("Starting adding prefix to dataset identifier column.")
        data_id = data[identifier_column]
        column_dtype = data_id.dtype
        if column_dtype == object:
            left_prefix_presence = data_id.str.startswith('l').all()
            right_prefix_presence = data_id.str.startswith('r').all()

            if left_prefix_presence or right_prefix_presence:
                self.logger.info("Column identifier prefix already existing.")
                data[identifier_column] = range(len(data))

        data[identifier_column] = data[identifier_column].apply(lambda x: "{}{}".format(prefix, x))

        self.logger.info("New prefix added successfully.")

        return data


class BlockingComponent(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_attribute_equivalence_blocker(self, A, B, block_attribute):

        self.logger.info("Applying attribute equivalence blocker on attribute {}.".format(block_attribute))
        # Instantiate attribute equivalence blocker object
        ab = em.AttrEquivalenceBlocker()
        # Use block_tables to apply blocking over two input tables.
        # FIXME: sto ipotizzando che i dataset di input abbiano lo stesso schema
        blocked_data = ab.block_tables(A, B,
                             l_block_attr=block_attribute, r_block_attr=block_attribute,
                             l_output_attrs=list(A.columns.values[1:]), r_output_attrs=list(B.columns.values[1:]),
                             l_output_prefix='ltable_', r_output_prefix='rtable_')

        return blocked_data


class EMComponent(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_em_rule_based_technique(self, A, B, blocked_data, rules):

        self.logger.info("Applying a rule matcher based on rules: {}.".format(rules))

        # create a boolean rule matcher
        brm = em.BooleanRuleMatcher()

        # generate a set of features
        feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)

        # add rules to boolean rule matcher
        for rule in rules:
            brm.add_rule(rule, feature_table)

        predictions = brm.predict(blocked_data, target_attr='predicted', append=True)

        #eval_result = em.eval_matches(predictions, 'gold', 'pred_label')
        #print("Precision: {}".format(eval_result["precision"]))
        #print("Recall: {}".format(eval_result["recall"]))

        return predictions

    def apply_em_machine_learning_technique(self, A, B, ground_truth):

        self.logger.info("Starting applying EM with a machine learning approach.")

        # Set the seed value
        seed = 0

        # [BEGIN] TRAIN AND TEST SETS CREATION -------------------------------------

        self.logger.info("Train and test sets creation.")
        # Split S into I an J
        train_test = em.split_train_test(ground_truth, train_proportion=0.7, random_state=0)
        train = train_test['train']
        test = train_test['test']
        self.logger.info("Train: {}".format(train.shape))
        self.logger.info("Test: {}".format(test.shape))

        # [END] TRAIN AND TEST SETS CREATION -----------------------------------

        # [BEGIN] CREATING A SET OF LEARNING-BASED MATCHERS --------------------

        self.logger.info(
            "Creating a set of ML-matchers: Decision Tree, SVM, Random Forest, Logistic Regression, Linear Regression.")
        # Create a set of ML-matchers
        dt = em.DTMatcher(name='DecisionTree', random_state=seed)
        svm = em.SVMMatcher(name='SVM', random_state=seed)
        rf = em.RFMatcher(name='RF', random_state=seed)
        lg = em.LogRegMatcher(name='LogReg', random_state=seed)
        ln = em.LinRegMatcher(name='LinReg')

        # [END] CREATING A SET OF LEARNING-BASED MATCHERS ----------------------

        # [BEGIN] CREATING FEATURES --------------------------------------------

        self.logger.info("Creating a set of features based on multiple similarity functions applied to attribute pairs.")
        feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)

        # [END] CREATING FEATURES ----------------------------------------------

        # [BEGIN] EXTRACTING FEATURE VECTORS -----------------------------------

        self.logger.info("Extracting feature vectors from train and test sets.")
        # Convert the train set into a set of feature vectors using the feature table
        train_feature_set = em.extract_feature_vecs(train, feature_table=feature_table, attrs_after='label',
                                                    show_progress=False)

        test_feature_set = em.extract_feature_vecs(test, feature_table=feature_table, attrs_after='label',
                                                   show_progress=False)

        def _fill_nan_in_feature_vectors(feature_vectors):

            # Check if the feature vectors contain missing values
            # A return value of True means that there are missing values
            null_presence = any(pd.notnull(feature_vectors))

            if null_presence:

                self.logger.debug("Filling missing values in feature vectors.")

                # Impute feature vectors with the mean of the column values.
                feature_vectors = em.impute_table(feature_vectors,
                                                    exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                                    strategy='mean')
            else:

                self.logger.debug("Feature vectors already clean (no missing values).")

            return feature_vectors

        train_feature_set = _fill_nan_in_feature_vectors(train_feature_set)
        test_feature_set = _fill_nan_in_feature_vectors(test_feature_set)

        # [END] EXTRACTING FEATURE VECTORS -------------------------------------

        # [BEGIN] SELECTING THE BEST MATCHER -----------------------------------

        self.logger.info("Selecting the best matcher using 5 fold cross validation.")
        matchers = [dt, rf, svm, ln, lg]
        result = em.select_matcher(matchers, table=train_feature_set,
                                   exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                   k=5,
                                   target_attr='label', metric_to_select_matcher='f1', random_state=seed)
        self.logger.info(result['cv_stats'])
        # print(result['drill_down_cv_stats']['precision'])
        # print(result['drill_down_cv_stats']['recall'])
        # print(result['drill_down_cv_stats']['f1'])
        best_matcher_id = result['cv_stats']["Average f1"].idxmax()
        self.logger.info("Best matcher: {}.".format(result['cv_stats'].iloc[best_matcher_id, :]["Matcher"]))

        best_matcher = matchers[best_matcher_id]

        # [END] SELECTING THE BEST MATCHER -------------------------------------

        # [BEGIN] TRAIN MODEL --------------------------------------------------
        self.logger.info("Training best matcher.")
        best_matcher.fit(table=train_feature_set, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                         target_attr='label')
        # [END] TRAIN MODEL ----------------------------------------------------

        # [BEGIN] MODEL PREDICTION ---------------------------------------------

        self.logger.info("Starting predicting.")
        predictions = best_matcher.predict(table=test_feature_set,
                                           exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                           target_attr='predicted',
                                           append=True, inplace=True)

        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        self.logger.info("Precision: {}".format(eval_result["precision"]))
        self.logger.info("Recall: {}".format(eval_result["recall"]))

        # [END] MODEL PREDICTION -----------------------------------------------

        return predictions


class ClusteringComponent(object):

    def __init__(self, matched_entities):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matched_entities = matched_entities

    def apply_connected_components_technique(self):

        self.logger.info("Applying connected components technique on matched entries.")

        matched_pred = self.matched_entities[self.matched_entities["predicted"] == 1]
        unique_entity1_ids = set([r for r in matched_pred["ltable_id"].unique()])
        unique_entity2_ids = set([r for r in matched_pred["rtable_id"].unique()])
        validation_nodes = unique_entity1_ids.union(unique_entity2_ids)

        matches = {}
        for i, row in matched_pred.iterrows():
            block_match = tuple(sorted((row["ltable_id"], row["rtable_id"])))
            matches[block_match] = 1
        self.logger.debug("Number of nodes in the graph: {}.".format(len(matches)))

        # create cluster of matches
        graph = nx.Graph()
        for node in validation_nodes:
            graph.add_node(node)
        for edge in matches:
            graph.add_edge(edge[0], edge[1])

        # find connected components
        connected_components = nx.connected_components(graph)

        return connected_components


class DataFusionComponent(object):
    def __init__(self, clustered_entries, dataset1, dataset2):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clustered_entries = clustered_entries
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def merge_all_entries(self):

        self.logger.info("Merging clustered entries.")
        ids_to_be_excluded = []
        final_entities = []
        for cluster in self.clustered_entries:

            if len(cluster) > 1:
                merged_entry = self._merge_entries_in_a_cluster(cluster)

                ids_to_be_excluded += list(cluster)
                final_entities.append(merged_entry)

        # building integrated dataset
        # FIXME: per il momento considero solo datasets in input che presentano lo stesso schema
        integrated_dataset = pd.concat([self.dataset1, self.dataset2])

        all_identifiers = set(integrated_dataset["id"].values)
        remaining_identifiers = list(all_identifiers.difference(set(ids_to_be_excluded)))
        integrated_dataset = integrated_dataset[integrated_dataset["id"].isin(remaining_identifiers)]

        new_entities = pd.DataFrame(data=final_entities)

        integrated_dataset = pd.concat([integrated_dataset, new_entities], sort=False)
        integrated_dataset.sort_values("id", inplace=True)
        integrated_dataset.reset_index(inplace=True, drop=True)
        integrated_dataset["id"] = range(len(integrated_dataset))

        self.logger.info("Integrated dataset: {}.".format(integrated_dataset.shape))

        return integrated_dataset

    def _merge_entries_in_a_cluster(self, entries_index, approach="first"):

        # ignored_entries = []
        merged_entry = None

        if approach == "first":
            for i, entry_index in enumerate(entries_index):
                if i > 0:
                    # ignored_entries.append(entry_index)
                    break
                else:

                    if entry_index.startswith("l"):
                        entry_index = entry_index.replace("l", "")
                        entry_index = int(entry_index)
                        merged_entry = self.dataset1.iloc[entry_index, :].to_dict()

                    elif entry_index.startswith("r"):

                        entry_index = entry_index.replace("r", "")
                        entry_index = int(entry_index)
                        merged_entry = self.dataset1.iloc[entry_index, :].to_dict()

                    else:
                        raise ValueError("Id doesn't start with 'l' or 'r'.")

        return merged_entry  # , ignored_entries


class TwoSourcesIntegrator(object):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_rule_based_integration(self, dataset1, dataset2, block_attribute, rules):

        self.logger.info("Integrating dataset {} and dataset {} with a rule-based approach.".format(dataset1, dataset2))

        # [BEGIN] LOAD DATASETS ------------------------------------------------

        data_loader = DataLoaderComponent()
        A = data_loader.load_dataset(dataset1)
        B = data_loader.load_dataset(dataset2)

        # [END] LOAD DATASETS --------------------------------------------------

        # [BEGIN] DATASET PREPROCESSING ----------------------------------------

        data_preprocessing = DataPreprocessingComponent()
        A = data_preprocessing.add_prefix_to_identifier_column(A, 'id', 'l')
        B = data_preprocessing.add_prefix_to_identifier_column(B, 'id', 'r')

        # [END] DATASET PREPROCESSING ------------------------------------------

        # [BEGIN] BLOCKING -----------------------------------------------------

        blocker = BlockingComponent()
        blocked_data = blocker.apply_attribute_equivalence_blocker(A, B, block_attribute)

        # [END] BLOCKING -------------------------------------------------------

        # [BEGIN] EM PROCESS ---------------------------------------------------

        em_component = EMComponent()
        predictions = em_component.apply_em_rule_based_technique(A, B, blocked_data, rules)

        # [END] EM PROCESS -----------------------------------------------------

        # [BEGIN] CLUSTERING ---------------------------------------------------

        clustering_component = ClusteringComponent(predictions)
        connected_components = clustering_component.apply_connected_components_technique()

        # [END] CLUSTERING -----------------------------------------------------

        # [BEGIN] DATA FUSION --------------------------------------------------

        # merge entries that belong to the same cluster
        fusion_component = DataFusionComponent(connected_components, A, B)
        integrated_dataset = fusion_component.merge_all_entries()

        # [END] DATA FUSION ----------------------------------------------------

        return integrated_dataset

    def apply_machine_learning_integration(self, dataset1, dataset2, ground_truth):

        self.logger.info("Integrating dataset {} and dataset {} with a machine learning approach.".format(dataset1,
                                                                                                          dataset2))

        # [BEGIN] LOAD DATASETS AND GROUND TRUTH -------------------------------

        data_loader = DataLoaderComponent()
        A, B, ground_truth = data_loader.load_magellan_datasets(dataset1, dataset2, ground_truth)

        # [END] LOAD DATASETS AND GROUND TRUTH ---------------------------------

        # [BEGIN] DATASET PREPROCESSING ----------------------------------------

        data_preprocessing = DataPreprocessingComponent()
        A = data_preprocessing.add_prefix_to_identifier_column(A, 'id', 'l')
        B = data_preprocessing.add_prefix_to_identifier_column(B, 'id', 'r')

        # [END] DATASET PREPROCESSING ------------------------------------------

        # [BEGIN] EM PROCESS ---------------------------------------------------

        em_component = EMComponent()
        predictions = em_component.apply_em_machine_learning_technique(A, B, ground_truth)

        # [END] EM PROCESS -----------------------------------------------------

        # [BEGIN] CLUSTERING ---------------------------------------------------

        clustering_component = ClusteringComponent(predictions)
        connected_components = clustering_component.apply_connected_components_technique()

        # [END] CLUSTERING -----------------------------------------------------

        # [BEGIN] DATA FUSION --------------------------------------------------

        # merge entries that belong to the same cluster
        fusion_component = DataFusionComponent(connected_components, A, B)
        integrated_dataset = fusion_component.merge_all_entries()

        # [END] DATA FUSION ----------------------------------------------------

        return integrated_dataset


class MultiSourcesIntegrator(object):

    def __init__(self, datasets):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.datasets = datasets
        self.num_datasets = len(datasets)

        self.output_directory = os.path.join(ROOT_DIR, "output")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def apply_rule_based_integration(self, block_attribute, rules):

        self.logger.info("Integrating multiple datasets: {}.".format(', '.join(self.datasets)))

        dataset1 = self.datasets[0]
        dataset2 = self.datasets[1]

        integrator = TwoSourcesIntegrator()
        integrated_dataset = integrator.apply_rule_based_integration(dataset1, dataset2, block_attribute, rules)

        if self.num_datasets > 2:

            for i in range(2, self.num_datasets):
                partial_integrated_dataset = "_partial_integration_{}.csv".format(i-1)
                partial_integrated_dataset_path = os.path.join(self.output_directory, partial_integrated_dataset)
                integrated_dataset.to_csv(partial_integrated_dataset_path, index=False)

                dataset = self.datasets[i]
                integrated_dataset = integrator.apply_rule_based_integration(partial_integrated_dataset_path, dataset,
                                                                             block_attribute, rules)

        integrated_dataset.to_csv(os.path.join(self.output_directory, "integrated_dataset.csv"), index=False)

        return integrated_dataset


if __name__ == '__main__':


    ####################################################################################################################
    # [OPTION A] RULE-BASED APPROACH
    datasets = ['data/source1.csv', 'data/source2.csv', 'data/source3.csv']
    block_attribute = "Citta"
    rules = [
        ["Nome_Nome_exm(ltuple, rtuple) == 1", "Cognome_Cognome_exm(ltuple, rtuple) == 1",
         "Citta_Citta_exm(ltuple, rtuple) == 1"],
    ]

    multi_integrator = MultiSourcesIntegrator(datasets)
    integrated_dataset = multi_integrator.apply_rule_based_integration(block_attribute, rules)
    print(integrated_dataset)
    ####################################################################################################################


    ####################################################################################################################
    # [OPTION B] MACHINE LEARNING APPROACH
    # dataset1 = 'dblp_demo.csv'
    # dataset2 = 'acm_demo.csv'
    # ground_truth = 'labeled_data_demo.csv'
    #
    # integrator = TwoSourcesIntegrator()
    # integrated_dataset = integrator.apply_machine_learning_integration(dataset1, dataset2, ground_truth)
    # print(integrated_dataset)
    ####################################################################################################################
