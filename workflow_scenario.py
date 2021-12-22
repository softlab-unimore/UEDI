import numpy as np
import pandas as pd
import os
import collections

from uedi.nlp.pre_processing import TextPreparation
from uedi.nlp.language_models import LanguageModel
from uedi.data_integration.data_preparation import DataPreparationComponent
from uedi.evaluation import ScenarioGenerator

from uedi.plot import plot_two_res


# Dataset Tokenizer
def attribute_tokenizer(df: pd.DataFrame, columns: list, join: bool = False):
    assert isinstance(df, pd.DataFrame)  #  check input DataFrame
    assert isinstance(columns, list)  #  check input columns

    res = df[columns]

    if join:
        res = np.array(res.astype(str).apply(lambda x: [' '.join(x)], axis=1).to_list())
    else:
        res = res.values

    return res


# Select Source 1, Source 2 and Source 3 from integrated dataset
def get_source(df, sid):
    cond = df['source'] == sid
    if cond.any():
        return df.loc[cond, :].iloc[0, :]
    else:
        return df.iloc[0, :]


def get_random(df):
    idx = np.random.randint(0, len(df))
    return df.iloc[idx, :]


def get_integrated_s1_s2(integrated_df, option=1):
    if option == 1:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x: get_source(x, sid=0))

    elif option == 2:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x: get_source(x, sid=1))
    else:
        integrated_s1_s2 = integrated_df.groupby('entity_id').apply(lambda x: get_random(x))

    return integrated_s1_s2


def get_s1_s2_s3(df, size_s1, size_s2, size_s3):
    print('{} entities in integrated dataframe with {} records'.format(len(df['entity_id'].unique()), len(df)))

    integrated_df = df.head(size_s1 * 2).copy()
    print('{} entities'.format(len(integrated_df['entity_id'].unique())))

    s1 = integrated_df[integrated_df['source'] == 0].reset_index()
    s2 = integrated_df[integrated_df['source'] == 1].reset_index()

    s1 = s1.iloc[:size_s1]
    s2 = s2.iloc[:size_s2]

    print('{} entities in s1 with {} records'.format(len(s1['entity_id'].unique()), len(s1)))
    print('{} entities in s2 with {} records'.format(len(s2['entity_id'].unique()), len(s2)))

    s3 = df.tail(size_s3).reset_index()
    print('{} entities in s3 with {} records'.format(len(s3['entity_id'].unique()), len(s3)))

    return s1, s2, s3


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


def test():
    # read integrated dataset
    df = pd.read_csv('data/restaurant/labeled_data.csv')

    # define target features
    features = ['NAME', 'ADDRESS']

    # generate three data sources
    size_s1 = 100
    size_s2 = 100
    size_s3 = 100

    # lm evaluation parameters
    mode = 'difference'
    resize = True

    s1, s2, s3 = get_s1_s2_s3(df, size_s1, size_s2, size_s3)

    # Get combined dataset

    # Scenario 1
    s1_s1_concat = pd.concat([s1, s1], ignore_index=True)

    # Scenario 2
    s1_s2_concat = pd.concat([s1, s2], ignore_index=True)
    s1_s2_match = get_integrated_s1_s2(s1_s2_concat, option=1)

    # Scenario 3
    s1_s3_concat = pd.concat([s1, s3], ignore_index=True)

    # Get records
    s1_records = attribute_tokenizer(s1, columns=features, join=True).squeeze()
    s2_records = attribute_tokenizer(s2, columns=features, join=True).squeeze()
    s3_records = attribute_tokenizer(s3, columns=features, join=True).squeeze()

    # Scenario 2 integrations
    s1_s2_match_records = attribute_tokenizer(s1_s2_match, columns=features, join=True).squeeze()
    s1_s2_concat_records = attribute_tokenizer(s1_s2_concat, columns=features, join=True).squeeze()

    # Scenario 1 integrations
    s1_s1_concat_records = attribute_tokenizer(s1_s1_concat, columns=features, join=True).squeeze()

    # Scenario 3 integrations
    s1_s3_concat_records = attribute_tokenizer(s1_s3_concat, columns=features, join=True).squeeze()

    # Scenarios 1 sources and integrations dataset
    sources = [s1_records, s1_records, s1_records]
    integrations = [s1_records, s1_records]
    integrations_concat = [s1_records, s1_s1_concat_records]

    res_scores_match, _, _ = lm_evaluation(sources, integrations, mode=mode)
    res_scores_concat, _, _ = lm_evaluation(sources, integrations_concat, mode=mode)

    plot_two_res(res_scores_match, res_scores_concat, resize=resize,
                 title='scenario 2 (same entity) with {} records in s1 and {} records in s2'.format(size_s1, size_s2))

    # Scenario 2
    sources = [s1_records, s1_records, s2_records]
    integrations = [s1_records, s1_s2_match_records]
    integrations_concat = [s1_records, s1_s2_concat_records]

    res_scores_match, _, _ = lm_evaluation(sources, integrations, mode=mode)
    res_scores_concat, _, _ = lm_evaluation(sources, integrations_concat, mode=mode)

    plot_two_res(res_scores_match, res_scores_concat, resize=resize,
                 title='scenario 2 (same entity) with {} records in s1 and {} records in s2'.format(size_s1, size_s2))

    # Scenario 3
    sources = [s1_records, s1_records, s3_records]
    integrations = [s1_records, s1_records]
    integrations_concat = [s1_records, s1_s3_concat_records]

    res_scores_match, _, _ = lm_evaluation(sources, integrations, mode=mode)
    res_scores_concat, _, _ = lm_evaluation(sources, integrations_concat, mode=mode)

    plot_two_res(res_scores_match, res_scores_concat, resize=resize,
                 title='scenario 2 (same entity) with {} records in s1 and {} records in s2'.format(size_s1, size_s2))

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


def compare_multi_lm(lms1, lms2, target_index=None):
    """
    This function compares the language models included in the list lms2 with the ones included in the list lms1.
    The following comparison schema is applied: the i-th elements of lms2 is compared with the first i+1 elements of
    lms1. If the user provides an index of interest (target_index parameter), only the target_index-th element of lms2
    is considered.

    :param lms1: iterable object containing some LanguageModel objects
    :param lms2: iterable object containing some LanguageModel objects
    :param target_index: optional index to focus the comparison
    :return: DataFrame object containing the results of the comparison
    """

    if not isinstance(lms1, collections.Iterable):
        raise TypeError("Wrong data type for parameter lms1. Only iterable data type is allowed.")

    if not isinstance(lms2, collections.Iterable):
        raise TypeError("Wrong data type for parameter lms2. Only iterable data type is allowed.")

    for lm1 in lms1:
        if not isinstance(lm1, LanguageModel):
            raise TypeError("Wrong data type for lms1 elements. Only LanguageModel data type is allowed.")

    for lm2 in lms2:
        if not isinstance(lm2, LanguageModel):
            raise TypeError("Wrong data type for lms2 elements. Only LanguageModel data type is allowed.")

    lms1 = list(lms1)
    lms2 = list(lms2)

    if target_index is not None:
        if not isinstance(target_index, int):
            raise TypeError("Wrong data type for parameter target_iteration. Only integer data type is allowed.")

    if target_index <= 0 or target_index > len(lms2) + 1:
        raise TypeError(
            "Wrong data value for parameter target_iteration. Only values in the range [1, {}] are allowed.".format(
                len(lms2) + 1))

    lm_cmp_results = []
    for lm2_id, lm2 in enumerate(lms2, 1):

        if target_index is not None:
            if lm2_id != target_index:
                continue

        for lm1_id, lm1 in enumerate(lms1):
            lm_cmp_res = compare_lm(lm1, lm2)
            res = {"source_id": lm1_id, "integration_id": lm2_id}
            res.update(lm_cmp_res)
            lm_cmp_results.append(res)

    lm_cmp_table = pd.DataFrame(lm_cmp_results)

    return lm_cmp_table

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

    # get detailed evaluation of language model comparison
    target_iteration = 2
    lm_cmp_table = compare_multi_lm(lm_sources, lm_integrations, target_iteration)

    return res_scores, lm_cmp_table


def main():
    # STEP 1: get data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data")
    exp_data_dir = os.path.join(root_dir, "data_for_experiments")
    lm_data_dir = os.path.join(exp_data_dir, "lm_data")
    user_data_dir = os.path.join(data_dir, "user_data")

    # EXAMPLE RESTAURANT
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

    # EXAMPLE BIKE
    # example = 'bike'
    # example_dir = os.path.join(data_dir, example)
    # # source 1
    # file1 = os.path.join(example_dir, "original", "bikedekho.csv")
    # source1_id = 0
    # # source 2
    # file2 = os.path.join(example_dir, "original", "bikewale.csv")
    # source2_id = 1
    # # integrated data
    # file3 = os.path.join(example_dir, "original", "labeled_data.csv")

    example = 'movie'
    # example_dir = os.path.join(data_dir, example)
    # # source 1
    # file1 = os.path.join(example_dir, "original", "rotten_tomatoes.csv")
    # source1_id = 0
    # # source 2
    # file2 = os.path.join(example_dir, "original", "imdb.csv")
    # source2_id = 1
    # # integrated data
    # file3 = os.path.join(example_dir, "original", "labeled_data.csv")

    # END STEP 1

    # STEP 2: prepare data
    # RESTAURANT
    data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3, "_id", "ltable._id",
                                              "rtable._id", "gold")
    # data_prep_comp = DataPreparationComponent(file1, source1_id, file2, source2_id, file3, "_id", "ltable.Id",
    #                                           "rtable.Id", "gold", data1_index_col="Id", data2_index_col="Id")

    # END STEP 2

    # define target features

    # restaurant example
    # features = ['NAME', 'ADDRESS']
    features = ["NAME", "RATING", "PHONENUMBER", "NO_OF_REVIEWS", "ADDRESS"]

    # bike example
    # features = ["bike_name", "city_posted", "km_driven", "color", "fuel_type", "price", "model_year", "owner_type",
    #             "url"]

    # movie example
    # features = ["Name", "Year", "Release Date", "Director", "Creator", "Actors", "Cast", "Language", "Country",
    #             "Duration", "RatingValue", "RatingCount", "ReviewCount", "Genre", "Filming Locations", "Description"]

    # lm evaluation parameters
    # mode = 'difference'
    mode = 'precision'
    resize = True

    # def prepare_pair(x):
    #     s, i = x.split("_")[2:]
    #     s = s.replace("s", "")
    #     i = i.replace("i", "")
    #
    #     return "{}-{}".format(i, s)
    #
    # # (OPTIONAL) get user data
    # user_data_file = os.path.join(user_data_dir, "summary.xlsx")
    # user_data_table = pd.read_excel(user_data_file)
    # user_data_table["scenario_id"] = user_data_table["Scenario"].apply(lambda x: x.split("_")[0])
    # user_data_table["case"] = user_data_table["Scenario"].apply(lambda x: x.split("_")[1])
    # user_data_table["pair"] = user_data_table["Scenario"].apply(lambda x: prepare_pair(x))

    # integration scenarios
    scenarios = ["duplicate", "source", "size", "info_type", "integration_result"]
    for scenario_id, scenario in enumerate(scenarios, 1):
        sources_A, integrations_A, sources_B, integrations_B = generate_scenarios(data_prep_comp, scenario)
        print("SOURCES A")
        for s in sources_A:
            print(s.get_data().shape)

        print("INTEGRATIONS A")
        for i in integrations_A:
            print(i.get_data().shape)

        print("SOURCES B")
        for s in sources_B:
            print(s.get_data().shape)

        print("INTEGRATIONS B")
        for i in integrations_B:
            print(i.get_data().shape)

        res_A, lm_cmp_table_A = run_sub_scenario(sources_A, integrations_A, features, mode)
        res_B, lm_cmp_table_B = run_sub_scenario(sources_B, integrations_B, features, mode)

        # # (OPTIONAL) save language model vocabs overlap
        # lm_data_file_name_A = os.path.join(lm_data_dir, "lm_metrics_scenario{}_caseA.csv".format(scenario_id))
        # lm_data_file_name_B = os.path.join(lm_data_dir, "lm_metrics_scenario{}_caseB.csv".format(scenario_id))
        # lm_cmp_table_A.to_csv(lm_data_file_name_A, index=False)
        # lm_cmp_table_B.to_csv(lm_data_file_name_B, index=False)

        # # (OPTIONAL) load user data
        # user_data = {}
        # scenario_string = "scenario{}".format(scenario_id)
        # scenario_user_data = user_data_table[user_data_table["scenario_id"] == scenario_string]
        # metric_names = ["Fmeasure", "Fbeta-measure"]
        #
        # for case in ["caseA", "caseB"]:
        #     scenario_user_data_case = scenario_user_data[scenario_user_data["case"] == case]
        #
        #     case_x = []
        #     case_y = []
        #     case_labels = []
        #     for metric_name in metric_names:
        #         case_x += list(scenario_user_data_case["{}X1X2".format(metric_name)].values)
        #         case_y += list(scenario_user_data_case["{}X2X1".format(metric_name)].values)
        #         short_metric_name = "F1"
        #         if metric_name == "Fbeta-measure":
        #             short_metric_name += "B"
        #
        #         case_labels += ["{} {}".format(item, short_metric_name) for item in scenario_user_data_case["pair"].values]
        #
        #     case_user_data = {"x": case_x, "y": case_y, "labels": case_labels}
        #
        #     user_data[case] = case_user_data
        #
        # plot_two_res_with_user_data(res_A, res_B, user_data, resize=resize, title='scenario {}'.format(scenario))

        plot_two_res(res_A, res_B, resize=resize, title='scenario {}'.format(scenario))


if __name__ == '__main__':
    print('workflow scenario')
    main()
