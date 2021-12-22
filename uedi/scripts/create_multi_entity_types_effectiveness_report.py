from uedi.utils.file_utilities import check_file_existence
from uedi.utils.general_utilities import check_parameter_type

import os
import pandas as pd


def get_effectiveness(filename: str):
    """
    This function measures the results effectiveness.

    :param filename: file where results are stored
    """

    check_parameter_type(filename, 'filename', str, 'string')
    check_file_existence(filename)

    data = pd.read_csv(filename)
    metric_info_by_scenario = ['mode', 'ngram', 'embed_manager', 'scenario']

    def _get_effectiveness(x):
        scenario = x['scenario'].values[0]

        if scenario == 1:
            acc = (x['dist(IM)'] < x['dist(IC)']).sum()
        elif scenario == 2:
            acc = (x['dist(IC)'] < x['dist(IM)']).sum()
        else:
            acc = ((x['dist(IP)'] < x['dist(IM)']) & (x['dist(IP)'] < x['dist(IC)'])).sum()

        return acc

    acc_by_metric_and_scenario = data.groupby(metric_info_by_scenario).apply(_get_effectiveness)

    return acc_by_metric_and_scenario


if __name__ == '__main__':
    results_file = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'OLD_multi-entity-types',
                                'Summary_Table.csv')
    eff_scores = get_effectiveness(results_file)
    print(eff_scores)