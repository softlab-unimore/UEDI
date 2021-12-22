import os
import pandas as pd


def create_report_table(data):
    target_data = data.loc[data['scenario'] == 1,:].squeeze()
    target_dist = target_data['score (x,y)']
    target_dist_ynorm = target_data['score (x,y norm)']
    target_norm_dist = target_data['norm score (x, y)']
    target_norm_dist_ynorm = target_data['norm score (x, y norm)']

    report_table = pd.DataFrame()
    report_table["dist movement"] = data['score (x,y)'] - target_dist
    report_table["dist ynorm movement"] = data['score (x,y norm)'] - target_dist_ynorm
    report_table["norm dist movement"] = data['norm score (x, y)'] - target_norm_dist
    report_table["norm dist ynorm movement"] = data['norm score (x, y norm)'] - target_norm_dist_ynorm

    report_table["dist movement (%)"] = (report_table["dist movement"] / target_dist) * 100
    report_table["dist ynorm movement (%)"] = (report_table["dist ynorm movement"] / target_dist_ynorm) * 100
    report_table["norm dist movement (%)"] = (report_table["norm dist movement"] / target_norm_dist) * 100
    report_table["norm dist ynorm movement (%)"] = (report_table[
                                                        "norm dist ynorm movement"] / target_norm_dist_ynorm) * 100
    report_table['changed_entities'] = data['changed_entities']

    return report_table


if __name__ == '__main__':

    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'entities-variation', 'NEW')

    results_sub_dirs = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                        os.path.isdir(os.path.join(results_dir, f))]

    results_files = [os.path.join(sub_dir, f) for sub_dir in results_sub_dirs for f in os.listdir(sub_dir) if
                     os.path.isfile(os.path.join(sub_dir, f)) and 'report' not in f]

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

    use_cases_reports = []
    for use_case in sorted_use_cases:
        df_use_case = all_df[all_df['use_case'] == use_case]

        approach_params = ['mode', 'ngram', 'embed_manager']
        group_data_by_approach = df_use_case.groupby(approach_params)

        reports_multi_approaches = []
        for approach_conf, data_by_approach in group_data_by_approach:
            approach = approach_conf[0]
            if 'emb' in approach:
                approach = approach_conf[2]
            approach = approach_map[approach]

            report_multi_approaches = create_report_table(data_by_approach)

            report_multi_approaches['approach'] = approach

            reports_multi_approaches.append(report_multi_approaches)

        use_case_report_data = pd.concat(reports_multi_approaches)

        use_case_pivot = use_case_report_data.pivot(index=['changed_entities'], columns=['approach'])

        use_case_pivot['use case'] = use_case

        use_cases_reports.append(use_case_pivot)

    report = pd.concat(use_cases_reports)
    report.to_csv(os.path.join(results_dir, "report.csv"))
