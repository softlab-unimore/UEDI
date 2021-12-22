import os
import pandas as pd
import numpy as np


def extract_intervals_by_feature(data, feature, interval_cols):
    extracted_data = {}
    for ix, row in data.iterrows():
        feature_val = float("{:.1f}".format(row[feature]))
        vals_map = {}
        for int_id, interval_col in enumerate(interval_cols, 1):
            vals_map[(feature_val, 'interval{}'.format(int_id))] = float("{:.1f}".format(row[interval_col]))
        extracted_data.update(vals_map)

    return extracted_data


def create_stats_table(data, datasets):
    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}
    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(use_cases_map) + 1)]

    first_level_cols = [float("{:.1f}".format(x)) for x in np.linspace(0.1, 1, 10)]
    second_level_cols = ["interval1", "interval2", "interval3"]
    index = pd.MultiIndex.from_product([first_level_cols, second_level_cols])
    stats_table = pd.DataFrame(data, columns=index)
    stats_table.index = datasets

    stats_table.index = stats_table.index.map(use_cases_map)
    stats_table = stats_table.reindex(index=sorted_use_cases)

    return stats_table


def create_min_tot_stats_tables(results_map):

    totality_table_data = []
    minimality_table_data = []
    norm_minimality_table_data = []
    datasets = []
    for file in results_map:
        file_relative_name = file.split(os.sep)[-1]
        dataset = file_relative_name.replace("_results.csv", "")
        data = results_map[file]
        datasets.append(dataset)

        totality_table_record = extract_intervals_by_feature(data[data['frac'] > 0], 'frac',
                                                             ['Ix interval1', 'Ix interval2', 'Ix interval3'])
        minimality_table_record = extract_intervals_by_feature(data[data['dup'] > 0], 'dup',
                                                               ['Iy interval1', 'Iy interval2', 'Iy interval3'])
        norm_minimality_table_record = extract_intervals_by_feature(data[data['dup'] > 0], 'dup',
                                                                    ['Iy normalized interval1',
                                                                     'Iy normalized interval2',
                                                                     'Iy normalized interval3'])

        totality_table_data.append(totality_table_record)
        minimality_table_data.append(minimality_table_record)
        norm_minimality_table_data.append(norm_minimality_table_record)

    tot_stats = create_stats_table(totality_table_data, datasets)
    min_stats = create_stats_table(minimality_table_data, datasets)
    norm_min_stats = create_stats_table(norm_minimality_table_data, datasets)

    return tot_stats, min_stats, norm_min_stats


def create_dist_stats_tables(results_map):

    distance_table_data = []
    distance_ynorm_table_data = []
    norm_distance_table_data = []
    norm_distance_ynorm_table_data = []
    datasets = []
    for file in results_map:
        file_relative_name = file.split(os.sep)[-1]
        dataset = file_relative_name.replace("_results.csv", "")
        data = results_map[file]
        datasets.append(dataset)

        dataset_dist_data = data[data['changed_entities'] > 0]
        distance_table_record = extract_intervals_by_feature(dataset_dist_data, 'changed_entities',
                                                             ['score (x,y) interval1', 'score (x,y) interval2',
                                                              'score (x,y) interval3'])
        distance_ynorm_table_record = extract_intervals_by_feature(dataset_dist_data, 'changed_entities',
                                                                   ['score (x,y norm) interval1',
                                                                    'score (x,y norm) interval2',
                                                                    'score (x,y norm) interval3'])
        norm_distance_table_record = extract_intervals_by_feature(dataset_dist_data, 'changed_entities',
                                                                  ['norm score (x, y) interval1',
                                                                   'norm score (x, y) interval2',
                                                                   'norm score (x, y) interval3'])
        norm_distance_ynorm_table_record = extract_intervals_by_feature(dataset_dist_data, 'changed_entities',
                                                                        ['norm score (x, y norm) interval1',
                                                                         'norm score (x, y norm) interval2',
                                                                         'norm score (x, y norm) interval3'])

        distance_table_data.append(distance_table_record)
        distance_ynorm_table_data.append(distance_ynorm_table_record)
        norm_distance_table_data.append(norm_distance_table_record)
        norm_distance_ynorm_table_data.append(norm_distance_ynorm_table_record)

    dist_stats = create_stats_table(distance_table_data, datasets)
    dist_ynorm_stats = create_stats_table(distance_ynorm_table_data, datasets)
    norm_dist_stats = create_stats_table(norm_distance_table_data, datasets)
    norm_dist_ynorm_stats = create_stats_table(norm_distance_ynorm_table_data, datasets)

    return dist_stats, dist_ynorm_stats, norm_dist_stats, norm_dist_ynorm_stats


if __name__ == '__main__':

    # metric = 'jaccard'
    metric = 'difference'

    # data_fusion_option = 'select_by_source'
    # data_fusion_option = 'random_records'
    data_fusion_option = 'random_attribute_values'

    exp_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'experiment-significance')
    min_tot_results_dir = os.path.join(exp_results_dir, 'minimality-totality', metric, data_fusion_option)
    distance_results_dir = os.path.join(exp_results_dir, 'entities-variation', metric, data_fusion_option)

    # minimality + totality
    min_tot_results_files = [os.path.join(min_tot_results_dir, f) for f in os.listdir(min_tot_results_dir) if
                             os.path.isfile(os.path.join(min_tot_results_dir, f)) and f.endswith(
                                 '.csv') and 'stats' not in f]
    min_tot_list_of_results = {f: pd.read_csv(f) for f in min_tot_results_files}

    # distance
    distance_results_files = [os.path.join(distance_results_dir, f) for f in os.listdir(distance_results_dir) if
                              os.path.isfile(os.path.join(distance_results_dir, f)) and f.endswith(
                                  '.csv') and 'stats' not in f]
    distance_list_of_results = {f: pd.read_csv(f) for f in distance_results_files}

    tot_stats, min_stats, norm_min_stats = create_min_tot_stats_tables(min_tot_list_of_results)
    dist_stats, dist_ynorm_stats, norm_dist_stats, norm_dist_ynorm_stats = create_dist_stats_tables(
        distance_list_of_results)

    tot_stats.to_csv(os.path.join(min_tot_results_dir, 'stats_tot_table.csv'))
    min_stats.to_csv(os.path.join(min_tot_results_dir, 'stats_min_table.csv'))
    norm_min_stats.to_csv(os.path.join(min_tot_results_dir, 'stats_norm_min_table.csv'))
    dist_stats.to_csv(os.path.join(distance_results_dir, 'stats_dist_table.csv'))
    dist_ynorm_stats.to_csv(os.path.join(distance_results_dir, 'stats_dist_ynorm_table.csv'))
    norm_dist_stats.to_csv(os.path.join(distance_results_dir, 'stats_norm_dist_table.csv'))
    norm_dist_ynorm_stats.to_csv(os.path.join(distance_results_dir, 'stats_norm_dist_ynorm_table.csv'))
