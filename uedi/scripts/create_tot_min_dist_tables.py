import os
import pandas as pd
import numpy as np


def create_final_dataframe(data, datasets):

    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}
    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(use_cases_map) + 1)]

    df = pd.DataFrame(data)
    df.index = datasets

    df.index = df.index.map(use_cases_map)
    df = df.reindex(index=sorted_use_cases)

    return df


def create_minimality_totality_tables(results_map):
    """
    This function creates the minimality and totality tables from the results obtained over multiple datastes.

    :param results_map: results grouped by dataset
    :return: totality and minimality tables
    """

    final_totality_table_data = []
    final_minimality_table_data = []
    final_normalized_minimality_table_data = []
    datasets = []
    for file in results_map:
        file_relative_name = file.split(os.sep)[-1]
        dataset = file_relative_name.replace("_results.csv", "")
        data = results_map[file]
        datasets.append(dataset)

        totality_table = data[data['frac'] > 0]
        totality_table_record = {}
        for ix, row in totality_table.iterrows():
            frac = float("{:.1f}".format(row['frac']))
            val = float("{:.3f}".format(row['Ix']))
            totality_table_record[frac] = val

        minimality_table = data[data['dup'] > 0]
        minimality_table_record = {}
        normalized_minimality_table_record = {}
        for ix, row in minimality_table.iterrows():
            dup = float("{:.1f}".format(row['dup']))
            val = float("{:.3f}".format(row['Iy']))
            norm_val = float("{:.3f}".format(row['Iy normalized']))
            minimality_table_record[dup] = val
            normalized_minimality_table_record[dup] = norm_val

        final_totality_table_data.append(totality_table_record)
        final_minimality_table_data.append(minimality_table_record)
        final_normalized_minimality_table_data.append(normalized_minimality_table_record)

    final_totality_table = create_final_dataframe(final_totality_table_data, datasets)
    final_minimality_table = create_final_dataframe(final_minimality_table_data, datasets)
    final_normalized_minimality_table = create_final_dataframe(final_normalized_minimality_table_data, datasets)

    return final_totality_table, final_minimality_table, final_normalized_minimality_table


def create_distance_tables(results_map):
    """
    This function creates the distance tables.

    :param results_map: results grouped by dataset
    :return: distance tables
    """

    distance_table_data = []
    distance_ynorm_table_data = []
    norm_distance_table_data = []
    norm_distance_ynorm_table_data = []
    xy_mean_table_data = []
    datasets = []
    for file in results_map:
        file_relative_name = file.split(os.sep)[-1]
        dataset = file_relative_name.replace("_results.csv", "")
        data = results_map[file]
        changed_entities = data['changed_entities']
        datasets.append(dataset)

        distance_table_record = {}
        distance_ynorm_table_record = {}
        norm_distance_table_record = {}
        norm_distance_ynorm_table_record = {}
        xy_mean_table_record = {}

        distances = data['score (x,y)']
        distances_ynorm = data['score (x,y norm)']
        norm_distances = data['norm score (x, y)']
        norm_distances_ynorm = data['norm score (x, y norm)']
        xy_mean = [np.mean(xy) for xy in zip(data['Ix'], data['Iy normalized'])]

        for ix, changed_entity_ratio in enumerate(changed_entities):
            distance_table_record[changed_entity_ratio] = float("{:.3f}".format(distances[ix]))
            distance_ynorm_table_record[changed_entity_ratio] = float("{:.3f}".format(distances_ynorm[ix]))
            norm_distance_table_record[changed_entity_ratio] = float("{:.3f}".format(norm_distances[ix]))
            norm_distance_ynorm_table_record[changed_entity_ratio] = float("{:.3f}".format(norm_distances_ynorm[ix]))
            xy_mean_table_record[changed_entity_ratio] = float("{:.3f}".format(xy_mean[ix]))

        distance_table_data.append(distance_table_record)
        distance_ynorm_table_data.append(distance_ynorm_table_record)
        norm_distance_table_data.append(norm_distance_table_record)
        norm_distance_ynorm_table_data.append(norm_distance_ynorm_table_record)
        xy_mean_table_data.append(xy_mean_table_record)

    distance_table = create_final_dataframe(distance_table_data, datasets)
    distance_ynorm_table = create_final_dataframe(distance_ynorm_table_data, datasets)
    norm_distance_table = create_final_dataframe(norm_distance_table_data, datasets)
    norm_distance_ynorm_table = create_final_dataframe(norm_distance_ynorm_table_data, datasets)
    xy_mean_table = create_final_dataframe(xy_mean_table_data, datasets)

    return distance_table, distance_ynorm_table, norm_distance_table, norm_distance_ynorm_table, xy_mean_table


if __name__ == '__main__':

    # metric = 'jaccard'
    metric = 'difference'

    data_fusion_option = 'select_by_source'
    # data_fusion_option = 'random_records'
    # data_fusion_option = 'random_attribute_values'

    min_tot_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'minimality-totality', 'NEW',
                                       metric, data_fusion_option)
    distance_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'entities-variation',
                                        'NEW', metric, data_fusion_option)

    # minimality + totality + distance
    min_tot_results_files = [os.path.join(min_tot_results_dir, f) for f in os.listdir(min_tot_results_dir) if
                             os.path.isfile(os.path.join(min_tot_results_dir, f)) and f.endswith(
                                 '.csv') and 'minimality' not in f and 'totality' not in f]
    min_tot_list_of_results = {f: pd.read_csv(f) for f in min_tot_results_files}

    distance_results_files = [os.path.join(distance_results_dir, f) for f in os.listdir(distance_results_dir) if
                              os.path.isfile(os.path.join(distance_results_dir, f)) and f.endswith(
                                 '.csv') and 'distance' not in f]
    distance_list_of_results = {f: pd.read_csv(f) for f in distance_results_files}

    totality_table, minimality_table, norm_minimality_table = create_minimality_totality_tables(min_tot_list_of_results)
    totality_table.to_csv(os.path.join(min_tot_results_dir, 'totality_table.csv'))
    norm_minimality_table.to_csv(os.path.join(min_tot_results_dir, 'norm_minimality_table.csv'))

    distance_table, distance_ynorm_table, norm_distance_table, norm_distance_ynorm_table, xy_mean_table = create_distance_tables(
        distance_list_of_results)
    distance_table.to_csv(os.path.join(distance_results_dir, 'distance_table.csv'))
    distance_ynorm_table.to_csv(os.path.join(distance_results_dir, 'distance_ynorm_table.csv'))
    norm_distance_table.to_csv(os.path.join(distance_results_dir, 'norm_distance_table.csv'))
    norm_distance_ynorm_table.to_csv(os.path.join(distance_results_dir, 'norm_distance_ynorm_table.csv'))
    xy_mean_table.to_csv(os.path.join(distance_results_dir, 'xy_mean_table.csv'))
