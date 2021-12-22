import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec


def create_axes_movements_report(report):
    select_cols = ['match ratio', 'non match ratio', 'changed records (%)', 'use case']
    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(report['use case'].unique()) + 1)]

    sel_report = report[select_cols]
    out_report = sel_report.pivot(index=['match ratio', 'non match ratio'], columns=['use case'])
    out_report.columns = out_report.columns.levels[1]
    out_report = out_report[sorted_use_cases]
    out_report = out_report.round(2)

    return out_report


def plot_measure_variation(report, measure, data_type):
    fig = plt.figure()
    fig.set_size_inches((18, 5))
    nrows, ncols = 2, 6
    plt.rcParams.update({'font.size': 16})

    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(report['use case'].unique()) + 1)]
    approach_order = ['global repr', 'jaccard', 'bleu score', 'word2vec', 'fasttext', 'glove']
    styles = ['o-', 'v-', '^-', 's-', 'D-', '*-']

    all_handles = []
    all_labels = []
    prec_ax = None
    for id, use_case in enumerate(sorted_use_cases, 1):
        if prec_ax is None:
            ax = plt.subplot(nrows, ncols, id)
        else:
            ax = plt.subplot(nrows, ncols, id, sharey=prec_ax)

        use_case_data = report[report['use case'] == use_case]

        index_cols = None
        if data_type == 'non_match':
            index_cols = ['non match ratio']
        elif data_type == 'match':
            index_cols = ['match ratio']
        use_case_data.set_index(index_cols, drop=True, inplace=True)

        sel_cols = []
        if measure == 'local':
            for approach in approach_order:
                sel_cols.append("('x movement', '{}')".format(approach))
        elif measure == 'global':
            # sel_cols = ["('y movement', 'global repr')"]
            sel_cols = ["('y norm movement', 'global repr')"]

        use_case_data = use_case_data[sel_cols]
        use_case_data.columns = [eval(c)[1] for c in use_case_data.columns]

        use_case_data.plot(rot=0, legend=False, ax=ax, title=use_case, style=styles, markersize=9)
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.xticks(xticks, [".{}".format(i) for i in range(0, 10, 2)] + ['1'])
        plt.xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels

        prec_ax = ax

    if measure == 'local':
        ytext = 'input representativeness variation'
    else:
        ytext = 'output representativeness variation'
    fig.text(0.005, 0.5, ytext, ha='center', va='center', rotation='vertical')
    x_axis_name = None
    if data_type == 'non_match':
        x_axis_name = 'unique entities error ratio'
    elif data_type == 'match':
        x_axis_name = 'shared entities error ratio'
    fig.text(0.5, 0.01, x_axis_name, ha='center', va='center', rotation='horizontal')

    if measure == 'local':
        approach_order[0] = 'our measure'
        fig.legend(all_handles, approach_order, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=7)

    plt.show()

    return fig


def create_striplot(data, ax, label, title):
    n = len(data)

    # initialize a plot
    ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, n)
    ax.set_yticks(data.iloc[:, 0].values)

    ax.set_yticklabels("")
    if title in ['U1', 'U7']:
        if label == 'x':
            ax.set_yticklabels(np.flip([str(x) for x in data.iloc[:, 0].values]))

    # define arrows
    arrow_starts = np.repeat(0, n)
    arrow_lengths = data.iloc[:, 1].values

    # add arrows to plot
    for i, index_val in enumerate(np.flip(data.iloc[:, 0].values)):

        if data.iloc[i, 2] == label:
            arrow_color = '#347768'
        else:
            arrow_color = '#6B273D'

        ax.arrow(arrow_starts[i],  # x start point
                 index_val,  # y start point
                 arrow_lengths[i],  # change in x
                 0,  # change in y
                 head_width=0.1,  # arrow head width
                 head_length=0.3,  # arrow head length
                 width=0.05,  # arrow stem width
                 fc=arrow_color,  # arrow fill color
                 ec=arrow_color)  # arrow edge color

    # format plot
    ax.axvline(x=0, color='0.1', ls='-', lw=1, zorder=0)  # add line at x=0
    ax.grid(axis='y', color='0.9')  # add a light grid
    ax.set_xlim(-1, 1)  # set x axis limits


def create_plot_for_measure_variations(x_data, y_data, title, fig, ax):

    sup_ax = plt.Subplot(fig, ax)
    sup_ax.set_title(title)
    sup_ax.axis('off')
    fig.add_subplot(sup_ax)

    sub_ax = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax)  # , wspace=0.1, hspace=0.1)

    sub_ax_1 = plt.Subplot(fig, sub_ax[0])
    create_striplot(x_data, sub_ax_1, 'x', title)
    fig.add_subplot(sub_ax_1)

    sub_ax_2 = plt.Subplot(fig, sub_ax[1])
    create_striplot(y_data, sub_ax_2, 'y', title)
    fig.add_subplot(sub_ax_2)


def plot_measure_variation_with_striplot(data, data_type):
    fig = plt.figure(figsize=(16, 4))
    ax = gridspec.GridSpec(2, 6)  # , wspace=0.2, hspace=0.2)
    plt.rcParams.update({'font.size': 14})

    sorted_use_cases = ['U{}'.format(i) for i in range(1, len(data['use case'].unique()) + 1)]

    for id, use_case in enumerate(sorted_use_cases, 1):
        use_case_data = data[data['use case'] == use_case]

        index_col = None
        index_name = None
        if data_type == 'non_match':
            index_col = 'non match ratio'
            index_name = 'wrong unique entities (%)'
        elif data_type == 'match':
            index_col = 'match ratio'
            index_name = 'wrong shared entities (%)'

        x_data = use_case_data[[index_col, "('x movement', 'global repr')", "('main norm movement', 'global repr')"]]
        x_data.columns = [index_name, 'delta totality', 'main movement']
        x_data = x_data[x_data[index_name] > 0]
        x_data.drop_duplicates(inplace=True)

        y_data = use_case_data[
            [index_col, "('y norm movement', 'global repr')", "('main norm movement', 'global repr')"]]
        y_data.columns = [index_name, 'delta minimality', 'main movement']
        y_data = y_data[y_data[index_name] > 0]
        y_data.drop_duplicates(inplace=True)

        create_plot_for_measure_variations(x_data, y_data, use_case, fig, ax[id - 1])

    fig.text(0.005, 0.5, index_name, ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.01, "variation on input (left) and output (right) representativeness", ha='center', va='center',
             rotation='horizontal')
    sns.despine(left=True, bottom=True)  # remove axes
    plt.show()
    return fig


# def plot_mix_axes_movements_report(report, cols):
#     fig = plt.figure()
#     fig.set_size_inches((18, 5))
#     nrows, ncols = 2, 6
#     plt.rcParams.update({'font.size': 16})
#
#     sorted_use_cases = ['U{}'.format(i) for i in range(1, len(report['use case'].unique()) + 1)]
#     approach_order = ['global repr', 'jaccard', 'bleu score', 'spacy', 'word2vec', 'fasttext', 'glove']
#
#     all_handles = []
#     all_labels = []
#     prec_ax = None
#     for id, use_case in enumerate(sorted_use_cases, 1):
#         if prec_ax is None:
#             ax = plt.subplot(nrows, ncols, id)
#         else:
#             ax = plt.subplot(nrows, ncols, id)
#
#         use_case_data = report[report['use case'] == use_case]
#         all_x = []
#         all_y = []
#         for approach in approach_order:
#             approach_x = use_case_data["('x movement (%)', '{}')".format(approach)]
#             all_x += list(approach_x.values)
#             approach_y = use_case_data["('y movement (%)', '{}')".format(approach)]
#             all_y += list(approach_y.values)
#             ax.plot(approach_x, approach_y, marker='o')
#
#         # FIXME: choose the right ranges
#         # xlim = ax.get_xlim()
#         # ylim = ax.get_ylim()
#         # lim_sup = np.max([xlim[1], ylim[1]])
#         # lim_inf = np.min([xlim[0], ylim[0]])
#         lim_sup = np.max(all_x + all_y)
#         lim_inf = np.min(all_x + all_y)
#         ax.set_xlim(lim_inf, lim_sup)
#         ax.set_ylim(lim_inf, lim_sup)
#
#         handles, labels = ax.get_legend_handles_labels()
#         all_handles += handles
#         all_labels += labels
#
#         prec_ax = ax
#
#     fig.text(0.01, 0.5, 'global measure variation (%)', ha='center', va='center', rotation='vertical')
#     legend = fig.legend(all_handles, approach_order, loc='lower center', fontsize=14,
#                         bbox_to_anchor=(0.5, -0.1), ncol=7)
#     plt.show()
#
#     return fig


if __name__ == '__main__':
    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-match-percentages', 'NEW')

    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                     os.path.isfile(os.path.join(results_dir, f))]

    match_report_file = 'match_report.csv'
    non_match_report_file = 'non_match_report.csv'

    match_report = pd.read_csv(os.path.join(results_dir, match_report_file))
    non_match_report = pd.read_csv(os.path.join(results_dir, non_match_report_file))

    # # create reports and plots
    # axes_movs_match_report = create_axes_movements_report(match_report)
    # axes_movs_match_plot = plot_measure_variation(match_report, 'global', 'match')
    #
    # axes_movs_non_match_report = create_axes_movements_report(non_match_report)
    # axes_movs_non_match_plot = plot_measure_variation(non_match_report, 'local', 'non_match')

    # create reports and plots
    match_striplot = plot_measure_variation_with_striplot(match_report, 'match')
    non_match_striplot = plot_measure_variation_with_striplot(non_match_report, 'non_match')

    # # save reports and plots
    # axes_movs_match_report.to_csv(os.path.join(results_dir, 'axes_movs_match_report.csv'))
    # axes_movs_match_plot.savefig(os.path.join(results_dir, 'axes_movs_match_plot.pdf'), bbox_inches='tight')
    #
    # axes_movs_non_match_report.to_csv(os.path.join(results_dir, 'axes_movs_non_match_report.csv'))
    # axes_movs_non_match_plot.savefig(os.path.join(results_dir, 'axes_movs_non_match_plot.pdf'), bbox_inches='tight')

    match_striplot.savefig(os.path.join(results_dir, 'match_striplot.pdf'), bbox_inches='tight')
    non_match_striplot.savefig(os.path.join(results_dir, 'non_match_striplot.pdf'), bbox_inches='tight')
