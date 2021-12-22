import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from cycler import cycler


def plot_tables(totality_table: pd.DataFrame, minimality_table: pd.DataFrame, distance_table: pd.DataFrame):
    fig = plt.figure()
    fig.set_size_inches(18, 4)
    nrows = 1
    ncols = 3

    cmap = plt.cm.get_cmap('RdYlBu', 12)
    styles = ['o-', 'v-', '^-', '<-', '>-', '8-', 's-', 'p-', 'D-', 'h-', 'H-', '*-']

    tot_ax = plt.subplot(nrows, ncols, 1)
    totality_table.plot(ax=tot_ax, legend=False, style=styles, linewidth=0.5, colormap=cmap, markersize=9)
    plt.xlabel('representation rate', fontsize=14)
    plt.xticks(range(10), ["{:.1f}".format(x) for x in np.linspace(0.1, 1, 10)], fontsize=14)
    plt.ylabel('input representativeness', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 1))
    all_handles, all_labels = tot_ax.get_legend_handles_labels()

    min_ax = plt.subplot(nrows, ncols, 2)
    minimality_table.plot(ax=min_ax, legend=False, style=styles, linewidth=0.5, colormap=cmap, markersize=9)
    plt.xlabel('duplication rate', fontsize=14)
    plt.xticks(range(10), ["{:.1f}".format(x) for x in np.linspace(0.1, 1, 10)], fontsize=14)
    plt.ylabel('output representativeness', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 1))

    dist_ax = plt.subplot(nrows, ncols, 3)
    distance_table = distance_table.iloc[1:len(distance_table), :]
    # distance_table = (distance_table * -1) + 1
    distance_table.plot(ax=dist_ax, legend=False, style=styles, linewidth=0.5, colormap=cmap, markersize=9)
    plt.xlabel('changed entities rate', fontsize=14)
    plt.xticks(range(10), ["{:.1f}".format(x) for x in np.linspace(0.1, 1, 10)], fontsize=14)
    plt.ylabel('representativeness distance', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0, 1))

    new_handles = [0 for i in range(len(all_handles))]
    new_labels = [0 for i in range(len(all_labels[:12]))]
    out_legend_grid_rows = 2
    out_legend_grid_cols = 6
    for i in range(len(all_handles)):
        if i < out_legend_grid_cols:
            new_index = i * out_legend_grid_rows
            new_handles[new_index] = all_handles[i]
            new_labels[new_index] = all_labels[i]
        else:
            div = i // out_legend_grid_cols
            new_index = (i - out_legend_grid_cols * div) * out_legend_grid_rows + div
            new_handles[new_index] = all_handles[i]
            new_labels[new_index] = all_labels[i]
    fig.legend(["U{}".format(i) for i in range(1, 13)], ncol=out_legend_grid_cols * out_legend_grid_rows, loc=8,
               fontsize=14, bbox_to_anchor=(0.5, -0.18))
    # plt.show()

    return fig


if __name__ == '__main__':

    # metric = 'jaccard'
    metric = 'difference'

    # data_fusion_option = 'select_by_source'
    # data_fusion_option = 'random_records'
    data_fusion_option = 'random_attribute_values'

    min_tot_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'minimality-totality', 'NEW',
                                       metric, data_fusion_option)
    distance_results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'entities-variation',
                                        'NEW', metric, data_fusion_option)

    totality_table_file_name = 'totality_table.csv'
    # minimality_table_file_name = 'minimality_table.csv'
    minimality_table_file_name = 'norm_minimality_table.csv'
    # distance_table_file_name = 'distance_table.csv'
    distance_table_file_name = 'distance_ynorm_table.csv'
    # distance_table_file_name = 'norm_distance_table.csv'
    # distance_table_file_name = 'norm_distance_ynorm_table.csv'
    # distance_table_file_name = 'xy_mean_table.csv'

    totality_table = pd.read_csv(os.path.join(min_tot_results_dir, totality_table_file_name), index_col=0)
    minimality_table = pd.read_csv(os.path.join(min_tot_results_dir, minimality_table_file_name), index_col=0)
    distance_table = pd.read_csv(os.path.join(distance_results_dir, distance_table_file_name), index_col=0)

    transposed_totality_table = totality_table.T
    transposed_minimality_table = minimality_table.T
    transposed_distance_table = distance_table.T

    cmaps = ['CMRmap_r', 'CMRmap', 'Accent', 'Accent_r', 'Dark2', 'Dark2_r', 'Greens', 'PRGn_r', 'Paired', 'Paired_r',
             'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'RdBu', 'RdBu_r', 'RdGy',
             'RdGy_r', 'RdYlBu']
    # for cmap_id in plt.colormaps():
    # for cmap_id in cmaps:
    #     print(cmap_id)
    #     cmap = plt.cm.get_cmap(cmap_id, 12)
    #     fig = plot_tables(transposed_totality_table, transposed_minimality_table, transposed_distance_table)
    #     fig.show()
    #     fig.savefig(os.path.join(min_tot_results_dir, '{}_tot_min_dist_plot_{}.pdf'.format(data_fusion_option, cmap_id)),
    #                     bbox_inches='tight')

    fig = plot_tables(transposed_totality_table, transposed_minimality_table, transposed_distance_table)

    fig.savefig(os.path.join(min_tot_results_dir, '{}_tot_min_dist_plot.pdf'.format(data_fusion_option)),
                bbox_inches='tight')
