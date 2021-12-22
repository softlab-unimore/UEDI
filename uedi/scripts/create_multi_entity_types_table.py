import pandas as pd
import os
from os import listdir
from os.path import isfile, join, abspath
import numpy as np


def extract_data(data, data_type):
    extracted_data = {}
    key = None
    if data_type == 'match':
        key = 'IM'
    elif data_type == 'concat':
        key = 'IC'
    elif data_type == 'perfect':
        key = 'IP'

    if len(data) > 0:
        x = float("{:.3f}".format(data['Ix'].values[0]))
        y = float("{:.3f}".format(data['Iy'].values[0]))
        y_norm = float("{:.3f}".format(data['Iy normalized'].values[0]))
        score = float("{:.3f}".format(data['score (x,y)'].values[0]))
        score_ynorm = float("{:.3f}".format(data['score (x,y norm)'].values[0]))
        norm_score = float("{:.3f}".format(data['norm score (x, y)'].values[0]))
        norm_score_ynorm = float("{:.3f}".format(data['norm score (x, y norm)'].values[0]))
    else:
        x = 0.0
        y = 0.0
        y_norm = 0.0
        score = 0.0
        score_ynorm = 0.0
        norm_score = 0.0
        norm_score_ynorm = 0.0

    i = (x, y)
    i_norm = (x, y_norm)

    extracted_data["{}x".format(key)] = x
    extracted_data["{}y".format(key)] = y
    extracted_data["{}y norm".format(key)] = y_norm
    extracted_data["{}".format(key)] = i
    extracted_data["{} norm".format(key)] = i_norm
    extracted_data["dist({})".format(key)] = score
    extracted_data["dist({} norm)".format(key)] = score_ynorm
    extracted_data["norm dist({})".format(key)] = norm_score
    extracted_data["norm dist({} norm)".format(key)] = norm_score_ynorm
    extracted_data["avg({})".format(key)] = "{:.3f}".format(np.average(i))
    extracted_data["avg({} norm)".format(key)] = "{:.3f}".format(np.average(i_norm))

    return extracted_data


def compress_scenario_data(x):
    out_row = x.iloc[0, :]

    m = x[x['data_type'] == 'match']
    c = x[x['data_type'] == 'concat']
    p = x[x['data_type'] == 'perfect']

    all = {}
    all.update(extract_data(m, 'match'))
    all.update(extract_data(c, 'concat'))
    all.update(extract_data(p, 'perfect'))

    for d in all:
        out_row[d] = all[d]

    if 'I' in out_row:
        del out_row['I']
    if 'D' in out_row:
        del out_row['D']
    del out_row['Ix']
    del out_row['Iy']
    del out_row['Iy normalized']
    del out_row['score (x,y)']
    del out_row['score (x,y norm)']
    del out_row['norm score (x, y)']
    del out_row['norm score (x, y norm)']
    del out_row['data_type']

    return out_row


if __name__ == '__main__':
    # metric = 'jaccard'
    metric = 'difference'

    results_dir = join(abspath(''), 'data', 'output', 'results', 'multi-entity-types', metric)

    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}

    # results_files = [join(results_dir, f) for f in listdir(results_dir) if
    #                  isfile(join(results_dir, f)) and f.endswith('.csv') and 'aggregated' in f]
    results_files = [join(results_dir, f) for f in listdir(results_dir) if
                     isfile(join(results_dir, f)) and f.endswith('.csv') and f != 'Summary_Table.csv']

    tables_for_summary = []
    for f in results_files:

        print(f)
        df = pd.read_csv(f)
        relative_name = f.split(os.sep)[-1]
        example_name_and_type = relative_name.replace("_results.csv", "")
        use_case = use_cases_map[example_name_and_type]
        d1 = "|D1|={}".format(df['D1'].values[0])
        d2 = "|D2|={}".format(df['D2'].values[0])
        d3 = "|D3|={}".format(df['D3'].values[0])
        d4 = "|D4|={}".format(df['D4'].values[0])
        v1 = "|V1|={}".format(df['V1'].values[0])
        v2 = "|V2|={}".format(df['V2'].values[0])
        v3 = "|V3|={}".format(df['V3'].values[0])
        v4 = "|V4|={}".format(df['V4'].values[0])
        df["example"] = "{} ({}, {}, {}, {}, {}, {}, {}, {})".format(use_case, d1, d2, d3, d4, v1, v2, v3, v4)

        new_df = df.groupby(['scenario', 'mode', 'ngram', 'embed_manager'], group_keys=False).apply(
            compress_scenario_data)
        new_df = new_df.reset_index(drop=True)

        tables_for_summary.append(new_df)

    if len(tables_for_summary) == 0:
        raise Exception("No files found.")

    summary_table = pd.concat(tables_for_summary)

    summary_table = summary_table[
        ['example', 'scenario', 'mode', 'ngram', 'embed_manager', 'IM', 'IM norm', 'IC', 'IC norm', 'IP', 'IP norm',
         'dist(IM)', 'dist(IC)', 'dist(IP)', 'dist(IM norm)', 'dist(IC norm)', 'dist(IP norm)', 'norm dist(IM)',
         'norm dist(IC)', 'norm dist(IP)', 'norm dist(IM norm)', 'norm dist(IC norm)', 'norm dist(IP norm)',
         'avg(IM)', 'avg(IC)', 'avg(IP)', 'avg(IM norm)', 'avg(IC norm)', 'avg(IP norm)']]
    print(summary_table.shape)
    print(summary_table.head(100))
    out_file = "Summary_Table.csv"
    summary_table["sort_col"] = summary_table["example"].apply(lambda x: int(x.split()[0].replace("U", "")))
    summary_table = summary_table.sort_values(by=['sort_col', 'scenario'])
    del summary_table["sort_col"]
    summary_table.to_csv(join(results_dir, out_file), index=False)
