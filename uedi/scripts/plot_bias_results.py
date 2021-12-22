import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_representativeness_bias(totality_table, minimality_table, distance_table):

    fig = plt.figure()
    fig.set_size_inches(18, 4)
    nrows = 1
    ncols = 3

    tot_ax = plt.subplot(nrows, ncols, 1)
    totality_table.plot(ax=tot_ax, legend=False, marker='o')
    plt.xlabel('sample size', fontsize=14)
    plt.xticks(totality_table.index.values, ["{:.1f}".format(x) for x in totality_table.index.values], fontsize=14)
    # plt.ylabel('input representativeness MSE', fontsize=14)
    plt.ylabel('input representativeness RMSE', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 0.1))
    # all_handles, all_labels = tot_ax.get_legend_handles_labels()

    min_ax = plt.subplot(nrows, ncols, 2)
    minimality_table.plot(ax=min_ax, legend=False, marker='o')
    plt.xlabel('sample size', fontsize=14)
    plt.xticks(minimality_table.index.values, ["{:.1f}".format(x) for x in minimality_table.index.values], fontsize=14)
    # plt.ylabel('output representativeness MSE', fontsize=14)
    plt.ylabel('output representativeness RMSE', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 0.1))

    dist_ax = plt.subplot(nrows, ncols, 3)
    distance_table.plot(ax=dist_ax, legend=False, marker='o')
    plt.xlabel('sample size', fontsize=14)
    plt.xticks(distance_table.index.values, ["{:.1f}".format(x) for x in distance_table.index.values], fontsize=14)
    # plt.ylabel('representativeness distance MSE', fontsize=14)
    plt.ylabel('representativeness distance RMSE', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 0.1))

    fig.legend(["U{}".format(i) for i in range(1, 13)], ncol=len(totality_table.columns.values), loc=8,
               fontsize=14,
               bbox_to_anchor=(0.5, -0.1))
    # plt.show()

    return fig


if __name__ == '__main__':
    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'bias')

    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                     os.path.isfile(os.path.join(results_dir, f)) and 'csv' in f]

    use_cases_map = {'Textual_Abt-Buy': 'U1', 'Dirty_DBLP-ACM': 'U7', 'Structured_iTunes-Amazon': 'U6',
                     'Structured_Amazon-Google': 'U2', 'Structured_DBLP-ACM': 'U8', 'Dirty_DBLP-GoogleScholar': 'U9',
                     'Dirty_Walmart-Amazon': 'U11', 'Dirty_iTunes-Amazon': 'U5', 'Structured_Beer': 'U3',
                     'Structured_DBLP-GoogleScholar': 'U10', 'Structured_Walmart-Amazon': 'U12',
                     'Structured_Fodors-Zagats': 'U4'}
    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(use_cases_map) + 1)]

    # regression_metric = 'squared_loss'
    regression_metric = 'rms_loss'
    totality = 'Ix'
    # minimality = 'Iy normalized'
    minimality = 'Iy'
    # distance = 'score (x,y norm)'
    distance = 'score (x,y)'

    totality_bias_data = []
    minimality_bias_data = []
    distance_bias_data = []
    for f in results_files:
        print(f)
        file_name = f.split(os.sep)[-1]
        dataset_name = '_'.join(file_name.split('_')[:2])
        use_case = use_cases_map[dataset_name]

        df = pd.read_csv(f)

        tot_bias = df[['sample_size', "{}({})".format(regression_metric, totality)]].copy()
        tot_bias['use_case'] = use_case
        totality_bias_data.append(tot_bias)

        min_bias = df[['sample_size', "{}({})".format(regression_metric, minimality)]].copy()
        min_bias['use_case'] = use_case
        minimality_bias_data.append(min_bias)

        dist_bias = df[['sample_size', "{}({})".format(regression_metric, distance)]].copy()
        dist_bias['use_case'] = use_case
        distance_bias_data.append(dist_bias)

    totality_bias = pd.concat(totality_bias_data)
    minimality_bias = pd.concat(minimality_bias_data)
    distance_bias = pd.concat(distance_bias_data)

    tot_bias_plot_data = totality_bias.pivot(index=['sample_size'], columns=['use_case'])
    tot_bias_plot_data.columns = list(tot_bias_plot_data.columns.levels[1])
    tot_bias_plot_data = tot_bias_plot_data[sorted_use_cases]

    min_bias_plot_data = minimality_bias.pivot(index=['sample_size'], columns=['use_case'])
    min_bias_plot_data.columns = list(min_bias_plot_data.columns.levels[1])
    min_bias_plot_data = min_bias_plot_data[sorted_use_cases]

    dist_bias_plot_data = distance_bias.pivot(index=['sample_size'], columns=['use_case'])
    dist_bias_plot_data.columns = list(dist_bias_plot_data.columns.levels[1])
    dist_bias_plot_data = dist_bias_plot_data[sorted_use_cases]

    bias_plot = plot_representativeness_bias(tot_bias_plot_data, min_bias_plot_data, dist_bias_plot_data)
    bias_plot.show()

    bias_plot.savefig(os.path.join(results_dir, 'bias_plot.pdf'), bbox_inches='tight')