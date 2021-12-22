from uedi.integration import Matcher
import py_entitymatching as em
import os
import collections
import numpy as np
import pandas as pd
from uedi.data_integration.data_preparation import convert_matching_pairs_to_integrated_dataset
from uedi.data_integration.data_fusion import DataFusionComponent
from uedi.nlp.language_models import LanguageModel
from uedi.nlp.pre_processing import TextPreparation
from uedi.plot import plot_two_res
from uedi.models.data_models import IntegratedDataset, DataSource


# LM Evaluation
def lm_evaluation(sources, integrations, mode='histogram'):
    res_scores = {}
    # fig, axes = plt.subplots()
    lm_integrations = []
    lm_sources_map = {}
    for i, integration in enumerate(integrations, 1):
        # tokenize current integrated dataset
        # docs_integration = [x.split() for x in integration]
        docs_integration = integration
        # create integrated dataset language model
        lm_integration = LanguageModel(n=1, mtype='mle')
        lm_integration.fit(docs_integration)
        lm_integrations.append(lm_integration)

        if mode == 'kl':
            # compute collection language model
            docs_collection = integration
            for j in range(i + 1):
                docs_collection = np.concatenate([docs_collection, sources[j]])

            # tokenize collection
            docs_collection = [x.split() for x in docs_collection]
            # create integrated dataset language model
            lm_collection = LanguageModel(n=1, mtype='mle')
            lm_collection.fit(docs_collection)

        # iterate over the past examples
        res_scores[i] = {
            'source': [],
            'integration': [],
        }
        for j in range(i + 1):
            # tokenize source dataset j
            # docs_source = [x.split() for x in sources[j]]
            docs_source = sources[j]
            # create source j language model
            lm_source = LanguageModel(n=1, mtype='mle')
            lm_source.fit(docs_source)
            if j not in lm_sources_map:
                lm_sources_map[j] = lm_source

            # ij = kl_language_model_lazy(lm_integration, lm_source)
            # ji = kl_language_model_lazy(lm_source, lm_integration)

            if mode == 'kl':
                ij = lm_integration.kl_smoothed(lm_source, lm_collection=lm_collection, lambd=0.8)
                ji = lm_source.kl_smoothed(lm_integration, lm_collection=lm_collection, lambd=0.8)

            elif mode == 'histogram':
                ij = lm_integration.histogram_intersection(lm_source)
                ji = lm_source.histogram_intersection(lm_integration)

            elif mode == 'overlap':
                ij = lm_integration.histogram_overlap(lm_source)
                ji = lm_source.histogram_overlap(lm_integration)

            elif mode == 'scaled_histogram':
                ij = lm_integration.scaled_histogram_intersection(lm_source)
                ji = lm_source.histogram_intersection(lm_integration)

            elif mode == 'difference':
                ji, ij = lm_source.histogram_difference(lm_integration)

            elif mode == 'precision':
                ji, ij = lm_source.precision(lm_integration)

            elif mode == 'recall':
                ji = lm_source.recall(lm_integration)
                ij = ji
            elif mode == 'fscore':
                rji = lm_source.recall(lm_integration)
                rij = rji
                pji, pij = lm_source.precision(lm_integration)

                ij = 2 * pij * rij / (pij + rij)
                ji = 2 * pji * rji / (pji + rji)

            else:
                raise ValueError('selected the wrong mod:', mode)

            res_scores[i]['source'].append(ji)
            res_scores[i]['integration'].append(ij)

            # axes.scatter(ji, ij, c='green')
            # axes.annotate('{}-{}'.format(i, j), (ji,ij),
            #               textcoords="offset points", xytext=(0,10), ha='center')

    lm_sources = [lm_sources_map[i] for i in range(len(lm_sources_map))]

    # axes.set_xlabel('source')
    # axes.set_xlabel('integration')
    # plt.show()
    return res_scores, lm_sources, lm_integrations


def run_sub_scenario(sources, integrations, features, mode):
    # clean source data
    clean_sources = []
    for source in sources:
        source_data = source.get_data()[features]
        source_attrs = source_data.columns.values
        clean_source_data = TextPreparation.convert_dataframe_to_text(source_data, source_attrs)
        clean_sources.append(clean_source_data)

    # clean integration data
    clean_integrations = []
    for integration in integrations:
        integration_data = integration.get_data()[features]
        integration_attrs = integration_data.columns.values
        clean_integration_data = TextPreparation.convert_dataframe_to_text(integration_data, integration_attrs)
        clean_integrations.append(clean_integration_data)

    # perform dataset language model comparison
    res_scores, lm_sources, lm_integrations = lm_evaluation(clean_sources, clean_integrations, mode=mode)

    return res_scores


def compare_two_matching_pairs(mp1, mp2, left_id, right_id, target):
    mp1 = mp1[[left_id, right_id, target]]
    mp2 = mp2[[left_id, right_id, target]]

    # i suppose that each matching pair data contain unique pair
    # join the two matching pair data on the pair identifiers
    concat_data = pd.merge(mp1, mp2, on=[left_id, right_id], suffixes=("_left", "_right"))
    left_attr = "{}_left".format(target)
    right_attr = "{}_right".format(target)
    concat_data["same_match"] = concat_data[left_attr] == concat_data[right_attr]

    different_match = concat_data[concat_data["same_match"] == False]
    print(different_match)

    exit(1)

def compare_lm(lm1, lm2):
    """
    This function compares two language models by computing simple counts about common and exclusive vocabularies.

    :param lm1: LanguageModel object
    :param lm2: LanguageModel object
    :return: dictionary containing the results of the comparison
    """

    if not isinstance(lm1, LanguageModel):
        raise TypeError("Wrong data type for parameter lm1. Only LanguageModel data type is allowed.")

    if not isinstance(lm2, LanguageModel):
        raise TypeError("Wrong data type for parameter lm2. Only LanguageModel data type is allowed.")

    data = {}

    lm1_vocabs = lm1.get_vocabs()
    lm2_vocabs = lm2.get_vocabs()
    common_vocabs = set(lm1_vocabs).intersection(set(lm2_vocabs))
    lm1_excl_vocabs = set(lm1_vocabs).difference(set(lm2_vocabs))
    lm2_excl_vocabs = set(lm2_vocabs).difference(set(lm1_vocabs))

    data["|Vl|"] = len(lm1_vocabs)
    data["|Vr|"] = len(lm2_vocabs)
    data["∑Vl"] = np.sum([lm1.model.counts[word] for word in lm1_vocabs])
    data["∑Vr"] = np.sum([lm2.model.counts[word] for word in lm2_vocabs])
    data["|∩|"] = len(common_vocabs)
    left_sum_common_vocabs = np.sum([lm1.model.counts[word] for word in common_vocabs])
    right_sum_common_vocabs = np.sum([lm2.model.counts[word] for word in common_vocabs])
    data["∑l(∩)"] = left_sum_common_vocabs
    data["∑r(∩)"] = right_sum_common_vocabs
    data["min(∑∩)"] = np.minimum(left_sum_common_vocabs, right_sum_common_vocabs)
    data["max(∑∩)"] = np.maximum(left_sum_common_vocabs, right_sum_common_vocabs)
    data["|Vl\Vr|"] = len(lm1_excl_vocabs)
    data["|Vr\Vl|"] = len(lm2_excl_vocabs)
    data["∑(Vl\Vr)"] = np.sum([lm1.model.counts[word] for word in lm1_excl_vocabs])
    data["∑(Vr\Vl)"] = np.sum([lm2.model.counts[word] for word in lm2_excl_vocabs])
    data["∑r(Vl)"] = np.sum([lm2.model.counts[word] for word in lm1_vocabs])

    return data

def compare_two_datasets(data1, data2, data1_pk, data2_pk):
    """
    This function compares two datasets by counting the number of common/exclusive rows. It is assumed that the record
    r1 (from data1) and the record r2 (from data1) are the same when they have the same unique code within their
    respective primary keys.

    :param data1: Pandas DataFrame containing the data of the first dataset
    :param data2: Pandas DataFrame containing the data of the second dataset
    :param data1_pk: primary key column of the first dataset
    :param data2_pk: primary key column of the second dataset
    :return: Python dictionary containing some metrics for measuring the "distance" between the two datasets
    """

    if not isinstance(data1, pd.DataFrame):
        raise TypeError("Wrong data type for parameter data1. Only Pandas DataFrame data type is allowed.")

    if not isinstance(data2, pd.DataFrame):
        raise TypeError("Wrong data type for parameter data2. Only Pandas DataFrame data type is allowed.")

    if not isinstance(data1_pk, str):
        raise TypeError("Wrong data type for parameter data1_pk. Only string data type is allowed.")

    if not isinstance(data2_pk, str):
        raise TypeError("Wrong data type for parameter data2_pk. Only string data type is allowed.")

    data1_cols = data1.columns.values
    data2_cols = data2.columns.values

    if data1_pk not in data1_cols:
        raise ValueError("Wrong data value for parameter data1_pk. Column {} not found in the DataFrame data1.".format(data1_pk))

    if data2_pk not in data2_cols:
        raise ValueError("Wrong data value for parameter data2_pk. Column {} not found in the DataFrame data2.".format(data2_pk))

    # FIXME: consider also the source column

    cmp_metrics = {}
    cmp_metrics["|A|"] = len(data1)
    cmp_metrics["|B|"] = len(data2)
    data1_set_ids = set(list(data1[data1_pk].values))
    data2_set_ids = set(list(data2[data2_pk].values))
    common_ids = data1_set_ids.intersection(data2_set_ids)
    left_ids = data1_set_ids.difference(data2_set_ids)
    right_ids = data2_set_ids.difference(data1_set_ids)
    cmp_metrics["|A∩B|"] = len(common_ids)
    cmp_metrics["|A-B|"] = len(left_ids)
    cmp_metrics["|B-A|"] = len(right_ids)
    cmp_metrics["%(A∩B)"] = "{:.2f}".format(len(common_ids) / (len(data1) + len(data2)))
    cmp_metrics["%(A-B)"] = "{:.2f}".format(len(left_ids) / len(data1))
    cmp_metrics["%(B-A)"] = "{:.2f}".format(len(right_ids) / len(data2))

    return cmp_metrics


def compare_two_integrated_datasets(integration1, integration2, integration1_pk, integration2_pk):
    """
    This function compares two integrated datasets by counting the number of common/exclusive entities and the
    "discrepancies" in the assignment of match/non-match labels between the first integrated dataset with respect the
    second one. The "discrepancies" in the assignment of match/non-match labels are measured with the concepts of
    "false negatives" and "false positives" defined at integrated dataset level (and not at matching pair level).
    Indeed, the metrics provided by this functions are not symmetric (i.e., changing the order of the input datasets the
    values will change). It is assumed that the record r1 (from integration1) and the record r2 (from integration2) are
    the same when they have the same unique code within their respective primary keys.

    :param integration1: IntegratedDataset object containing the data of the first integrated dataset
    :param integration2: IntegratedDataset object containing the data of the target integrated dataset
    :param integration1_pk: the primary key column in the first integrated dataset
    :param integration2_pk: the primary key column in the second integrated dataset
    :return: Python dictionary containing some metrics for the comparison of the two integrated datasets
    """

    if not isinstance(integration1, IntegratedDataset):
        raise TypeError("Wrong data type for parameter integration1. Only IntegratedDataset data type is allowed.")

    if not isinstance(integration2, IntegratedDataset):
        raise TypeError("Wrong data type for parameter integration2. Only IntegratedDataset data type is allowed.")

    if not isinstance(integration1_pk, str):
        raise TypeError("Wrong data type for parameter integration1_pk. Only string data type is allowed.")

    if not isinstance(integration2_pk, str):
        raise TypeError("Wrong data type for parameter integration2_pk. Only string data type is allowed.")

    data1 = integration1.get_data()
    data2 = integration2.get_data()
    data1_cols = data1.columns.values
    data2_cols = data2.columns.values

    if integration1_pk not in data1_cols:
        raise ValueError(
            "Wrong data value for parameter integration1_pk. Column {} not found in the DataFrame data1.".format(integration1_pk))

    if integration2_pk not in data2_cols:
        raise ValueError(
            "Wrong data value for parameter integration2_pk. Column {} not found in the DataFrame data2.".format(integration2_pk))

    cmp_metrics = compare_two_datasets(data1, data2, integration1_pk, integration2_pk)

    data1_non_match = data1[data1["match"] == 0]
    data1_non_match_ids = set(list(data1_non_match[integration1_pk].values))
    data1_match = data1[data1["match"] == 1]
    data1_match_ids = set(list(data1_match[integration1_pk].values))
    data2_non_match = data2[data2["match"] == 0]
    data2_non_match_ids = set(list(data2_non_match[integration2_pk].values))
    data2_match = data2[data2["match"] == 1]
    data2_match_ids = set(list(data2_match[integration2_pk].values))
    cmp_metrics["|A=|"] = len(data1_match)
    cmp_metrics["%(A=)"] = "{:.2f}".format(len(data1_match) / len(data1))
    cmp_metrics["|A≠|"] = len(data1_non_match)
    cmp_metrics["%(A≠)"] = "{:.2f}".format(len(data1_non_match) / len(data1))
    cmp_metrics["|B=|"] = len(data2_match)
    cmp_metrics["%(B=)"] = "{:.2f}".format(len(data2_match) / len(data2))
    cmp_metrics["|B≠|"] = len(data2_non_match)
    cmp_metrics["%(B≠)"] = "{:.2f}".format(len(data2_non_match) / len(data2))
    cmp_metrics["%(A=/B=)"] = "{:.2f}".format(len(data1_match) / len(data2_match))
    cmp_metrics["%(A≠/B≠)"] = "{:.2f}".format(len(data1_non_match) / len(data2_non_match))

    # get the number of "false negatives" (N.B.: this metric is defined over the integrated dataset and not over the
    # matching pairs)
    # the "false negatives" of data1 are the entities in data1 marked as non-match that are not included in data2 as
    # non-match
    data1_fn = data1_non_match_ids.difference(data2_non_match_ids)
    data1_fn_entities = data1[data1[integration1_pk].isin(data1_fn)]
    assert len(data1_fn) == len(data1_fn_entities)
    cmp_metrics["|A(fn)|"] = len(data1_fn)
    cmp_metrics["%(A≠(fn))"] = "{:.2f}".format(len(data1_fn) / len(data1_non_match)) if len(data1_non_match) > 0 else 0

    data1_match_and_fn = pd.concat([data1_match, data1_fn_entities])

    # get the number of "false positives" (N.B.: this metric is defined over the integrated dataset and not over the
    # matching pairs)
    # the "false positives" of data1 are the entities in data1 marked as match that are included in data2 as non-match
    data1_fp = data1_match_ids.intersection(data2_non_match_ids)
    data1_fp_entities = data1[data1[integration1_pk].isin(data1_fp)]
    assert len(data1_fp) == len(data1_fp_entities)
    cmp_metrics["|A(fp)|"] = len(data1_fp)
    cmp_metrics["%(A=(fp))"] = "{:.2f}".format(len(data1_fp) / len(data1_match)) if len(data1_match) > 0 else 0

    data1_non_match_and_fp = pd.concat([data1_non_match, data1_fp_entities])

    import pprint
    pprint.pprint(cmp_metrics)

    # exit(1)

    return cmp_metrics, data1_match_and_fn, data1_non_match_and_fp



def get_rule_matcher_integrations(m, multi_rules, type_data="All"):
    """
    This function applies multiple rule-based matchers based on the user-provided rules and returns the integrated
    datasets obtained by applying these rules.

    :param m: Matcher object that will apply the user-provided rules in order to discover matches
    :param multi_rules: the rules to be applied to the original datasets in order to create multiple integrated datasets
    :return: original datasets and list of integrated datasets
    """

    if not isinstance(m, Matcher):
        raise TypeError("Wrong data type for parameter matcher. Only Matcher data type is allowed.")

    if not isinstance(multi_rules, collections.Iterable):
        raise TypeError("Wrong data type for parameter multi_rules. Only iterable data type is allowed.")

    for rules in multi_rules:
        if not isinstance(rules, collections.Iterable):
            raise TypeError("Wrong data type for multi_rules elements. Only iterable data type is allowed.")

        for rule in rules:
            if not isinstance(rule, collections.Iterable):
                raise TypeError("Wrong data type for single rule in multi_rules. Only iterable data type is allowed.")

            for cond in rule:
                if not isinstance(cond, str):
                    raise TypeError("Wrong data type for single cond in multi_rules. Only string data type is allowed.")

    integrated_datasets = []
    original_data1, original_data2 = m.get_data()
    matching_results = m.apply_multi_rule_matchers(multi_rules)

    data_sources = []
    for matching_result in matching_results:
        match_pairs = matching_result[0]
        match_scores = matching_result[1]

        data1, data2, integration_data = convert_matching_pairs_to_integrated_dataset(original_data1, 0, "id",
                                                                                      original_data2, 1, "id",
                                                                                      match_pairs, "_id", "ltable_id",
                                                                                      "rtable_id", "pred_label")

        # print(integration_data.get_data()["entity_id"].value_counts())

        # random data fusion
        data_fusion_comp = DataFusionComponent(integration_data)
        merged_dataset = data_fusion_comp.select_random_records(random_seed)
        mdata = merged_dataset.get_data()
        mdata_id_col = merged_dataset.get_id_col()
        mdata_source_col = merged_dataset.get_source_id_col()
        mdata_entity_col = merged_dataset.get_entity_label_col()

        # compare the current integrated dataset with the perfect integration (i.e., the gold dataset)
        cmp_metrics, mdata_match, mdata_non_match = compare_two_integrated_datasets(merged_dataset, gold_dataset, "id", "id")

        integrated_dataset = None
        if type_data == "match":
            integrated_dataset = IntegratedDataset(mdata_match, mdata_id_col, mdata_source_col, mdata_entity_col)

        elif type_data == "non-match":
            integrated_dataset = IntegratedDataset(mdata_non_match, mdata_id_col, mdata_source_col, mdata_entity_col)

        elif type_data == "All":
            integrated_dataset = merged_dataset.copy()

        integrated_dataset.set_effectiveness_scores(match_scores)

        integrated_datasets.append(integrated_dataset)

        idata = integrated_dataset.get_data()
        left_entities = idata[idata["source"] == 0]["id"].values
        right_entities = idata[idata["source"] == 1]["id"].values
        # filter original data sources
        d1 = data1.get_data()
        d2 = data2.get_data()
        data1_content = d1[d1["id"].isin(left_entities)]
        data2_content = d2[d2["id"].isin(right_entities)]
        # create final data sources and their integrated version
        data1 = DataSource(data1_content, 0)
        data2 = DataSource(data2_content, 1)

        data_sources.append([data1, data2])

    return data_sources, integrated_datasets


def get_ml_matcher_integrations(m, model_name, features_confs, type_data="All"):
    """
    This function applies multiple times the same ML matcher using multiple feature configurations.

    :param m: Matcher object
    :param model_name: name of ML model to be used for discovering matching pairs
    :param features_confs: feature configurations to be tested
    :return: original datasets and list of integrated datasets
    """

    integrated_datasets = []
    original_data1, original_data2 = m.get_data()
    matching_results = m.apply_ml_matcher_with_different_features(model_name, features_confs)

    mps = []
    data_sources = []
    for matching_result in matching_results:
        match_pairs = matching_result[0]
        match_scores = matching_result[1]
        mps.append(match_pairs)

        data1, data2, integration_data = convert_matching_pairs_to_integrated_dataset(original_data1, 0, "id",
                                                                                      original_data2, 1, "id",
                                                                                      match_pairs, "_id", "ltable_id",
                                                                                      "rtable_id", "pred_label")

        # random data fusion
        data_fusion_comp = DataFusionComponent(integration_data)
        merged_dataset = data_fusion_comp.select_random_records(random_seed)
        mdata = merged_dataset.get_data()
        mdata_id_col = merged_dataset.get_id_col()
        mdata_source_col = merged_dataset.get_source_id_col()
        mdata_entity_col = merged_dataset.get_entity_label_col()

        # compare the current integrated dataset with the perfect integration (i.e., the gold dataset)
        cmp_metrics, mdata_match, mdata_non_match = compare_two_integrated_datasets(merged_dataset, gold_dataset, "id", "id")

        integrated_dataset = None
        if type_data == "match":
            integrated_dataset = IntegratedDataset(mdata_match, mdata_id_col, mdata_source_col, mdata_entity_col)

        elif type_data == "non-match":
            integrated_dataset = IntegratedDataset(mdata_non_match, mdata_id_col, mdata_source_col, mdata_entity_col)

        elif type_data == "All":
            integrated_dataset = merged_dataset.copy()

        integrated_dataset.set_effectiveness_scores(match_scores)


        integrated_datasets.append(integrated_dataset)

        idata = integrated_dataset.get_data()
        left_entities = idata[idata["source"] == 0]["id"].values
        right_entities = idata[idata["source"] == 1]["id"].values
        # filter original data sources
        d1 = data1.get_data()
        d2 = data2.get_data()
        data1_content = d1[d1["id"].isin(left_entities)]
        data2_content = d2[d2["id"].isin(right_entities)]
        # create final data sources and their integrated version
        data1 = DataSource(data1_content, 0)
        data2 = DataSource(data2_content, 1)

        data_sources.append([data1, data2])

    # compare_two_matching_pairs(mps[0], mps[1], "ltable_id", "rtable_id", "pred_label")
    # compare_two_integrated_datasets(integrated_datasets[0], integrated_datasets[1], "id", "id", ['title', 'authors', 'venue', 'year'])

    return data_sources, integrated_datasets


def create_gold_dataset(A, B, S, random_seed):
    _, _, gold_data = convert_matching_pairs_to_integrated_dataset(A, 0, "id", B, 1, "id", S, "_id", "ltable_id",
                                                                   "rtable_id", "label")
    # random data fusion
    data_fusion_comp = DataFusionComponent(gold_data)
    gold_dataset = data_fusion_comp.select_random_records(random_seed)

    return gold_dataset


if __name__ == '__main__':
    # Get the datasets directory
    datasets_dir = em.get_install_path() + os.sep + 'datasets'

    path_A = datasets_dir + os.sep + 'dblp_demo.csv'
    path_B = datasets_dir + os.sep + 'acm_demo.csv'
    path_labeled_data = datasets_dir + os.sep + 'labeled_data_demo.csv'

    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # datasets_dir = os.path.join(root_dir, "sample_data", "examples")
    # path_A = os.path.join(datasets_dir, "tableA.csv")
    # path_B = os.path.join(datasets_dir, "tableA.csv")
    # path_labeled_data = os.path.join(datasets_dir, "gt.csv")

    A = em.read_csv_metadata(path_A, key='id')
    B = em.read_csv_metadata(path_B, key='id')
    features = ['title', 'authors', 'venue', 'year']

    # Load the pre-labeled data
    S = em.read_csv_metadata(path_labeled_data,
                             key='_id',
                             ltable=A, rtable=B,
                             fk_ltable='ltable_id', fk_rtable='rtable_id')
    true_labels = S["label"].values
    random_seed = 24

    # transform the ground truth into an integrated dataset
    gold_dataset = create_gold_dataset(A, B, S, random_seed)

    matcher = Matcher(A, B, S, random_state=random_seed)
    train_proportion = 0.5
    matcher.split_train_test(train_proportion)

    model_type = "Rule"
    # model_type = "ML"

    if model_type == "Rule":            # rule-based matcher
        model_name = model_type

        matcher_rules1 = [
            ['authors_authors_lev_sim(ltuple, rtuple) > 0.9']
        ]
        matcher_rules2 = [
            ['title_title_lev_sim(ltuple, rtuple) > 0.4', 'year_year_exm(ltuple, rtuple) == 1'],
            ['authors_authors_lev_sim(ltuple, rtuple) > 0.4']
        ]
        multi_matcher_rules = [matcher_rules1, matcher_rules2]

        sources, integrations = get_rule_matcher_integrations(matcher, multi_matcher_rules, type_data="match")

    else:                               # ml-based matcher

        model_name = "DecisionTree"
        # model_name = "SVM"
        # model_name = "RF"
        # model_name = "LogReg"
        # model_name = "LinReg"
        # model_name = "All"
        # print(matcher.get_feature_table())
        drop_features = [list(range(6,15))]
        sources, integrations = get_ml_matcher_integrations(matcher, model_name, drop_features, gold_dataset)

    # MACRO ANALYSIS
    results = []
    scores = []
    mode = 'difference'
    for index, integration in enumerate(integrations):

        # get the effectiveness scores associated to the current integrated dataset
        eval_res = integration.get_effectiveness_scores()
        f1 = "{:.2f}".format(eval_res['f1'])
        prec = "{:.2f}".format(eval_res['precision'])
        recall = "{:.2f}".format(eval_res['recall'])
        accuracy = "{:.2f}".format(eval_res['accuracy'])
        eval_map = {'f1': f1, 'p': prec, 'r': recall, 'acc': accuracy}
        print(eval_map)

        res = run_sub_scenario(sources[index], [integration], features, mode)
        results.append(res)
        scores.append(eval_map)

    resize = True
    plot_two_res(results[0], results[1], resize=resize, title='{} ({}, {})'.format(model_name, scores[0], scores[1]))

    # # MICRO ANALYSIS
    #
    # metric_name = "Jaccard"
    # profilers = []
    # for index, integration in enumerate(integrations):
    #     profiler = IntegrationProfiler(metric_name)
    #
    #     run_scenario(metric_name, sources[index], [integration], profiler, attrs=features)
    #
    #     profilers.append(profiler)
    #
    # # plot profiler result comparison
    # profilers_labels = ["Sub-scenario A", "Sub-scenario B"]
    # plot_multi_profiler_results(profilers, profilers_labels)
