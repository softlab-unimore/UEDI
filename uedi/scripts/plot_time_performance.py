import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'time-performance')

    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                     os.path.isfile(os.path.join(results_dir, f)) and f.endswith('.csv')]

    approach_map = {'difference': 'representativeness', 'jaccard_difference': 'jaccard', 'bleu_difference': 'bleu score',
                    'SpacyEmbeddingManager_lg': 'spacy', 'GensimEmbeddingManager_word2vec': 'embedding',
                    'GensimEmbeddingManager_fasttext': 'fasttext', 'GensimEmbeddingManager_glove': 'glove'}
    # approach_order = ['global repr', 'jaccard', 'bleu score', 'spacy', 'word2vec', 'fasttext', 'glove']
    approach_order = ['representativeness', 'jaccard', 'bleu score', 'embedding']

    results = pd.concat([pd.read_csv(f) for f in results_files])

    plot_data = []
    conf_params = ['scenario', 'mode', 'ngram', 'embed_manager']
    group_res_by_conf = results.groupby(conf_params)
    for conf_val, res_by_conf in group_res_by_conf:

        first_row = res_by_conf.iloc[0, :].copy()

        # plot_data_row = {'scenario': first_row['scenario'], 'data_size': first_row['data_size']}
        plot_data_row = {'data size': first_row['data_size']}

        approach = first_row['mode']
        if 'emb' in approach:
            approach = first_row['embed_manager']
        approach = approach_map[approach]
        if approach in ['spacy', 'fasttext', 'glove']:
            continue
        plot_data_row['approach'] = approach

        plot_data_row["time mean"] = np.mean(res_by_conf['total_time'].values)
        plot_data_row["time std"] = np.std(res_by_conf['total_time'].values)

        plot_data.append(plot_data_row)

    plot_table = pd.DataFrame(plot_data)
    reformat_plot_table = plot_table.pivot(index='data size', columns=['approach'])
    mean_times = reformat_plot_table['time mean'][approach_order]
    std_times = reformat_plot_table['time std'][approach_order]

    thresh_48h = 172800
    for col in ['jaccard', 'bleu score']:
        mean_times[col] = mean_times[col].fillna(thresh_48h)
    for col in ['embedding']:
        mean_times[col] = mean_times[col].fillna(0.2)

    plt.rcParams.update({'font.size': 18})
    mean_times.plot(logy=True, kind='bar', rot=0, figsize=(12, 5), width=0.8)
                    #yerr=std_times.values.T, error_kw=dict(ecolor='k'), alpha=0.5)
    approach_order[0] = 'our measure'
    plt.legend(approach_order, loc=8, bbox_to_anchor=(0.5, -0.35), ncol=4)
    plt.ylabel('Time (s)')
    plt.axhline(y=thresh_48h, linewidth=2, color="red", linestyle="--", zorder=0)
    text = plt.annotate('48h', (0, thresh_48h), textcoords="offset points", xytext=(-30, -20), ha='center')
    text.set_fontsize(18)
    plt.xticks(range(6), ['1K', '10K', '50K', '100K', '500K', '1M'])

    for i, (label, value) in enumerate(zip(mean_times.index.values, mean_times['embedding'].values)):
        if value == 0.2:
            plt.scatter(i + 0.3, 0.4, marker='X', color='r', s=500)
    plt.ylim(0.1, 250000)

    # plt.show()
    plt.savefig(os.path.join(results_dir, 'time_performance_plot.pdf'), bbox_inches='tight')
