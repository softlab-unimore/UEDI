import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_measure_variation(report):
    # cmap = plt.cm.get_cmap('RdYlBu', 6)
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
        use_case_data = use_case_data[approach_order]

        use_case_data.plot(rot=0, legend=False, ax=ax, title=use_case, style=styles, markersize=9)#, colormap=cmap)
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.xticks(xticks, [".{}".format(i) for i in range(0,10,2)] + ['1'])
        plt.xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        all_handles += handles
        all_labels += labels

        prec_ax = ax

    # fig.text(0.01, 0.5, 'distance variation (%)', ha='center', va='center', rotation='vertical')
    fig.text(0.005, 0.5, 'representativeness distance variation', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'changed entities', ha='center', va='center', rotation='horizontal')
    approach_order[0] = "our measure"
    fig.legend(all_handles, approach_order, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=7)

    plt.show()

    return fig


if __name__ == '__main__':

    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'entities-variation', 'NEW')

    report_file = 'report.csv'

    report = pd.read_csv(os.path.join(results_dir, report_file), header=[0, 1], index_col=0)

    # distance_measure = 'dist movement (%)'
    # distance_measure = 'dist ynorm movement (%)'
    # distance_measure = 'norm dist movement (%)'
    # distance_measure = 'norm dist ynorm movement (%)'

    # distance_measure = 'dist movement'
    distance_measure = 'dist ynorm movement'
    # distance_measure = 'norm dist movement'
    # distance_measure = 'norm dist ynorm movement'

    use_case_info = report[['use case']]
    use_case_info.columns = ['use case']
    seletect_report = pd.concat([report[distance_measure], use_case_info], axis=1)

    # create the plot
    report_plot = plot_measure_variation(seletect_report)

    # save the plot
    report_plot.savefig(os.path.join(results_dir, 'entity_variation_report_plot.pdf'), bbox_inches='tight')
