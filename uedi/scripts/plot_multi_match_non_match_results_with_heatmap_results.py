import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from uedi.utils.general_utilities import check_parameter_type, check_cols_in_dataframe


def compute_heatmap(data: pd.DataFrame):
    """

    """

    check_parameter_type(data, 'data', pd.DataFrame, 'Pandas DataFrame')

    conf_params = ['mode', 'ngram', 'embed_manager']
    check_cols_in_dataframe(data, conf_params)

    group_data_by_conf = data.groupby(conf_params)

    for conf_val, data_by_conf in group_data_by_conf:
        conf = zip(conf_params, conf_val)
        conf_str = ','.join(["{}={}".format(c[0], c[1]) for c in conf])

        vals = {}
        for ix, row in data_by_conf.iterrows():
            key = "{}-{}".format(row['match_ratio'], row['non_match_ratio'])
            val = (row['Ix'], row['Iy'])
            vals[key] = val

        order = ['1.0-1.0', '1.0-0.5', '1.0-0.0', '0.5-0.0', '0.5-0.5', '0.5-1.0', '0.0-1.0', '0.0-0.5', '0.0-0.0']
        sort_vals = [vals[x] for x in order]

        mat = np.zeros((len(sort_vals), len(sort_vals)))
        bool_mat = np.zeros((len(sort_vals), len(sort_vals)) , dtype=bool)
        for i in range(len(sort_vals)):
            for j in range(len(sort_vals)):
                if i == j:
                    continue
                mat[i,j] = float("{:.2f}".format((sort_vals[j][1] - sort_vals[i][1]) / (sort_vals[j][0] - sort_vals[i][0])))
                bool_mat[i,j] = bool(np.argmax(np.array([abs(sort_vals[j][0] - sort_vals[i][0]), abs(sort_vals[j][1] - sort_vals[i][1])])))

        fig, ax = plt.subplots()
        im = ax.imshow(bool_mat)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(order)))
        ax.set_yticks(np.arange(len(order)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(order)
        ax.set_yticklabels(order)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(order)):
            for j in range(len(order)):
                text = ax.text(j, i, mat[i, j],
                               ha="center", va="center", color="w")

        # plt.pcolor(bool_mat, cmap='RdYlGn')
        # im = ax.pcolor(mat, cmap=bool_mat, vmin=0, vmax=1, edgecolors='black')
        # cbar = fig.colorbar(im)

        ax.set_title("Metric variation by changing match and non-match recalls.")
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    results_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'results', 'multi-match-percentages', 'final')

    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if
                     os.path.isfile(os.path.join(results_dir, f)) and f.endswith('.csv')]

    for f in results_files:
        if 'Dirty_DBLP-ACM_results' not in f:
            continue
        print(f)
        data = pd.read_csv(f)
        compute_heatmap(data)