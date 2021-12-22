import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_table(table: pd.DataFrame):
    fig = plt.figure()
    fig.set_size_inches((18, 4.5))
    nrows, ncols = 2, 6
    plt.rcParams.update({'font.size': 16})

    table = table

    all_handles = []
    all_labels = []
    datasets = table.index.values
    for id, dataset in enumerate(datasets, 1):
        ax = plt.subplot(nrows, ncols, id)
        tot_table_single_data = table.loc[[dataset], :].T
        tot_table_single_data.index.names = ['sample dimension', 'intervals']
        tot_table_single_data.reset_index(inplace=True)
        pivot_tot_table_single_data = tot_table_single_data.pivot(index='sample dimension', columns='intervals')
        pivot_tot_table_single_data[(dataset, 'interval2')] = pivot_tot_table_single_data[(dataset, 'interval2')] - \
                                                              pivot_tot_table_single_data[(dataset, 'interval1')]
        pivot_tot_table_single_data[(dataset, 'interval3')] = pivot_tot_table_single_data[(dataset, 'interval3')] - \
                                                              (pivot_tot_table_single_data[(dataset, 'interval2')] + \
                                                               pivot_tot_table_single_data[(dataset, 'interval1')])
        pivot_tot_table_single_data[(dataset, 'interval1')] *= 100
        pivot_tot_table_single_data[(dataset, 'interval2')] *= 100
        pivot_tot_table_single_data[(dataset, 'interval3')] *= 100
        # r = (id - 1) // 6
        # c = (id - 1) % 6
        pivot_tot_table_single_data.plot(kind='bar', stacked=True, rot=0, legend=False, ax=ax, fontsize=14,
                                         title=dataset)
        plt.xticks(range(10), [".{}".format(i) for i in range(1, 10)] + [1])
        plt.yticks([0, 25, 50, 75, 100])
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels

    legend = fig.legend(all_handles, ['interval1: mean ± std', 'interval2: mean ± 2*std', 'interval3: mean ± 3*std'],
                        loc='upper center', fontsize=14, bbox_to_anchor=(0.5, 0.05), ncol=3)
    # legend.get_title().set_fontsize('14')
    fig.text(0.01, 0.5, 'Percentage (%)', ha='center', va='center', rotation='vertical')
    plt.show()

    return fig


if __name__ == '__main__':

    # metric = 'jaccard'
    metric = 'difference'

    # data_fusion_option = 'select_by_source'
    # data_fusion_option = 'random_records'
    data_fusion_option = 'random_attribute_values'

    experiment_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'experiment-significance')
    min_tot_results_dir = os.path.join(experiment_results_dir, 'minimality-totality', metric, data_fusion_option)
    dist_results_dir = os.path.join(experiment_results_dir, 'entities-variation', metric, data_fusion_option)

    stats_tot_table_file_name = 'stats_tot_table.csv'
    # stats_min_table_file_name = 'stats_min_table.csv'
    stats_min_table_file_name = 'stats_norm_min_table.csv'

    # stats_dist_table_file_name = 'stats_dist_table.csv'
    stats_dist_table_file_name = 'stats_dist_ynorm_table.csv'
    # stats_dist_table_file_name = 'stats_norm_dist_table.csv'
    # stats_dist_table_file_name = 'stats_norm_dist_ynorm_table.csv'

    # first_level_cols = [float("{:.1f}".format(x)) for x in np.linspace(0.1, 1, 10)]
    # second_level_cols = ["interval1", "interval2", "interval3"]
    # index = pd.MultiIndex.from_product([first_level_cols, second_level_cols])

    # read totality, minimality and distance tables
    stats_tot_table = pd.read_csv(os.path.join(min_tot_results_dir, stats_tot_table_file_name), index_col=0,
                                  header=[0, 1])
    stats_tot_table.columns = pd.MultiIndex.from_tuples(stats_tot_table.columns)
    stats_min_table = pd.read_csv(os.path.join(min_tot_results_dir, stats_min_table_file_name), index_col=0,
                                  header=[0, 1])
    stats_min_table.columns = pd.MultiIndex.from_tuples(stats_min_table.columns)
    stats_dist_table = pd.read_csv(os.path.join(dist_results_dir, stats_dist_table_file_name), index_col=0,
                                   header=[0, 1])
    stats_dist_table.columns = pd.MultiIndex.from_tuples(stats_dist_table.columns)

    # plot data
    stats_tot_plot = plot_table(stats_tot_table)
    stats_min_plot = plot_table(stats_min_table)
    stats_dist_plot = plot_table(stats_dist_table)

    # save plots
    stats_tot_plot.savefig(os.path.join(min_tot_results_dir, 'stats_tot_plot.pdf'), bbox_inches='tight')
    stats_min_plot.savefig(os.path.join(min_tot_results_dir, 'stats_min_plot.pdf'), bbox_inches='tight')
    stats_dist_plot.savefig(os.path.join(dist_results_dir, 'stats_dist_plot.pdf'), bbox_inches='tight')
