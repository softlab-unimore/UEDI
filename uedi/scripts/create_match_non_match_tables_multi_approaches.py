import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def create_report_table(data):
    target_data = data.loc[data['scenario'] == 1,:].squeeze()
    target_x = target_data['Ix']
    target_y = target_data['Iy']
    target_ynorm = target_data['Iy normalized']
    target_integration_dim = target_data['num_pred_integrated_entities']

    report_table = pd.DataFrame()
    report_table["x movement"] = data['Ix'] - target_x
    report_table["y movement"] = data['Iy'] - target_y
    report_table["y norm movement"] = data['Iy normalized'] - target_ynorm
    report_table['x movement (%)'] = (report_table["x movement"] / target_x) * 100
    report_table['y movement (%)'] = (report_table["y movement"] / target_y) * 100
    report_table['y norm movement (%)'] = (report_table["y norm movement"] / target_ynorm) * 100
    axes_names = ['x', 'y']
    report_table['main movement'] = report_table.apply(lambda x:
        axes_names[int(np.argmax((abs(x["x movement"]), abs(x["y movement"]))))], axis=1)
    report_table['main norm movement'] = report_table.apply(lambda x:
        axes_names[int(np.argmax((abs(x["x movement"]), abs(x["y norm movement"]))))], axis=1)
    report_table['#integration records'] = data['num_pred_integrated_entities']
    report_table['#changed records'] = data['num_pred_integrated_entities'] - target_integration_dim
    report_table['changed records (%)'] = (report_table['#changed records'] / target_integration_dim) * 100
    report_table['match ratio'] = data['match_ratio']
    report_table['non match ratio'] = data['non_match_ratio']

    return report_table


if __name__ == '__main__':

    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-match-percentages', 'NEW')

    results_sub_dirs = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                        os.path.isdir(os.path.join(results_dir, f))]

    results_files = [os.path.join(sub_dir, f) for sub_dir in results_sub_dirs for f in os.listdir(sub_dir) if
                     os.path.isfile(os.path.join(sub_dir, f))]

    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}
    approach_map = {'difference': 'global repr', 'jaccard_difference': 'jaccard', 'bleu_difference': 'bleu score',
                    'SpacyEmbeddingManager_lg': 'spacy', 'GensimEmbeddingManager_word2vec': 'word2vec',
                    'GensimEmbeddingManager_fasttext': 'fasttext', 'GensimEmbeddingManager_glove': 'glove'}
    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(use_cases_map) + 1)]

    all_data = []
    for f in results_files:
        print(f)
        file_name = f.split(os.sep)[-1]
        dataset_name = '_'.join(file_name.split('_')[:2])
        use_case = use_cases_map[dataset_name]

        df = pd.read_csv(f)
        df['use_case'] = use_case
        all_data.append(df)

    all_df = pd.concat(all_data)

    use_cases_match_reports = []
    use_cases_non_match_reports = []
    for use_case in sorted_use_cases:
        df_use_case = all_df[all_df['use_case'] == use_case]

        approach_params = ['mode', 'ngram', 'embed_manager']
        group_data_by_approach = df_use_case.groupby(approach_params)

        approaches_match_reports = []
        approaches_non_match_reports = []
        for approach_conf, data_by_approach in group_data_by_approach:
            approach = approach_conf[0]
            if 'emb' in approach:
                approach = approach_conf[2]
            approach = approach_map[approach]

            match_scenarios = [1, 8, 9, 10, 11, 12]
            non_match_scenarios = [1, 3, 4, 5, 6, 7]

            match_results = data_by_approach[data_by_approach['scenario'].isin(match_scenarios)]
            non_match_results = data_by_approach[data_by_approach['scenario'].isin(non_match_scenarios)]

            match_report = create_report_table(match_results)
            non_match_report = create_report_table(non_match_results)

            match_report['approach'] = approach
            non_match_report['approach'] = approach

            approaches_match_reports.append(match_report)
            approaches_non_match_reports.append(non_match_report)

        use_case_match_data = pd.concat(approaches_match_reports)
        use_case_non_match_data = pd.concat(approaches_non_match_reports)

        fixed_columns = ['match ratio', 'non match ratio', '#integration records', '#changed records',
                         'changed records (%)']

        use_case_match_fixed_data = use_case_match_data[fixed_columns]
        use_case_match_fixed_data = use_case_match_fixed_data.drop_duplicates()
        use_case_match_data = use_case_match_data.drop(fixed_columns[2:], axis=1)
        use_case_match_pivot = use_case_match_data.pivot(index=['match ratio', 'non match ratio'], columns=['approach'])
        use_case_match_report = pd.merge(use_case_match_pivot, use_case_match_fixed_data, left_index=True,
                                         right_on=['match ratio', 'non match ratio'])

        use_case_non_match_fixed_data = use_case_non_match_data[fixed_columns]
        use_case_non_match_fixed_data = use_case_non_match_fixed_data.drop_duplicates()
        use_case_non_match_data = use_case_non_match_data.drop(fixed_columns[2:], axis=1)
        use_case_non_match_pivot = use_case_non_match_data.pivot(index=['match ratio', 'non match ratio'],
                                                                 columns=['approach'])
        use_case_non_match_report = pd.merge(use_case_non_match_pivot, use_case_non_match_fixed_data, left_index=True,
                                             right_on=['match ratio', 'non match ratio'])

        use_case_match_report['use case'] = use_case
        use_case_non_match_report['use case'] = use_case

        use_cases_match_reports.append(use_case_match_report)
        use_cases_non_match_reports.append(use_case_non_match_report)

    match_report = pd.concat(use_cases_match_reports)
    match_report.to_csv(os.path.join(results_dir, "match_report.csv"), index=False)

    non_match_report = pd.concat(use_cases_non_match_reports)
    non_match_report.to_csv(os.path.join(results_dir, "non_match_report.csv"), index=False)
