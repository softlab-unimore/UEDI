import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from uedi.utils.general_utilities import check_parameter_type
from uedi.utils.file_utilities import create_dir
import os


class PlotManager(object):
    """
    This class implements a plot manager.
    """

    def __init__(self):
        """
        This method initializes the state of the plot manager.
        """
        self.figs = []
        self.figs_names = []
        self.plot_dir = os.path.join(os.path.abspath(''), 'data', 'output', 'plots')

    def plot(self):
        """
        This method plots the data.
        """
        pass

    def save_plots(self, file_prefix: str):
        """
        This method saves the plots to disk.

        :param file_prefix: the prefix to be added to plot file names
        """

        check_parameter_type(file_prefix, 'file_prefix', str, 'string')
        create_dir(self.plot_dir)

        for ix, fig in enumerate(self.figs):
            fig_name = self.figs_names[ix]
            file_name = os.path.join(self.plot_dir, "{}_{}".format(file_prefix, fig_name))
            fig.savefig(file_name)


class SimplePlotManager(PlotManager):

    def __init__(self, data: pd.DataFrame, plot_dir: str):

        check_parameter_type(data, 'data', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(plot_dir, 'plot_dir', str, 'string')

        if not os.path.exists(plot_dir):
            raise ValueError("Directory not found.")

        super().__init__()

        self.data = data
        self.plot_dir = plot_dir

    def plot(self):

        data = self.data.fillna('None')
        conf_params = ['mode', 'ngram', 'embed_manager']
        group_data_by_conf = data.groupby(conf_params)

        for conf_val, data_by_conf in group_data_by_conf:
            conf = zip(conf_params, conf_val)
            conf_str = ','.join(["{}={}".format(c[0], c[1]) for c in conf])

            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=((5, 5)))

            SimplePlotManager.plot_data(data_by_conf, ax)

            # fig.suptitle(conf_str)
            plt.show()

            self.figs.append(fig)
            self.figs_names.append("{}_plot.pdf".format(conf_str))

    @staticmethod
    def plot_data(data: pd.DataFrame, ax):

        for row_ix, row in data.iterrows():
            row_ix = int(row_ix)
            row = row.to_dict()
            x = row['Ix']
            y = row['Iy']

            ax.scatter(x, y, c='blue', s=60)
            if 'datasets' in row:
                label = row['datasets']
                text = ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
                text.set_fontsize(12)
            ax.tick_params(labelsize=18)

        # ax.set_xlim(left=-0.05, right=1.05)
        # ax.set_ylim(bottom=-0.05, top=1.05)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = min(xlim[0], ylim[0])
        ax.set_xlim(lim, 1.05)
        ax.set_ylim(lim, 1.05)

        ax.set_xlabel('source ($r_{D→I}$)', fontsize=24)
        ax.set_ylabel('integration ($r_{I→D}$)', fontsize=24)
        ax.grid()


class MultiEntityTypesPlotManager(PlotManager):
    """
    This class implements a plot manager for the results produced by the MultiEntityTypesEvaluator.
    """

    def __init__(self, data: pd.DataFrame, multi_subplots: bool = False, plot_dir: str = None):
        """
        This method initializes the state of the plot manager.

        :param data: Pandas DataFrame containing the data to plot.
        :param multi_subplots: boolean flag that indicates the number of subplots per scenario to be used.
        :param plot_dir: directory where to save the plots
        """
        check_parameter_type(data, 'data', pd.DataFrame, 'Pandas DataFrame')
        check_parameter_type(multi_subplots, 'multi_subplots', bool, 'boolean')
        check_parameter_type(plot_dir, 'plot_dir', str, 'string', optional_param=True)

        super().__init__()

        self.data = data
        self.multi_subplots = multi_subplots
        if plot_dir is not None:
            self.plot_dir = plot_dir
        else:
            self.plot_dir = os.path.join(self.plot_dir, 'multi-entity-types')

    def plot(self):
        """
        This method plots the data.
        """
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)

        scenarios = list(self.data['scenario'].unique())

        for scenario in scenarios:
            scenario_data = self.data[self.data['scenario'] == scenario]
            scenario_data = scenario_data.reset_index()
            self.plot_scenario_results(scenario_data, scenario)

    def plot_scenario_results(self, data: pd.DataFrame, scenario: int):
        """
        This method plots the data related to a specific scenario.

        :param data: Pandas DataFrame containing the data related to a specific scenario
        :param scenario: identifier of the considered scenario
        """

        data = data.fillna('None')
        conf_params = ['mode', 'ngram', 'embed_manager']
        group_data_by_conf = data.groupby(conf_params)

        for conf_val, data_by_conf in group_data_by_conf:
            conf = zip(conf_params, conf_val)
            conf_str = ','.join(["{}={}".format(c[0], c[1]) for c in conf])
            data_types = list(data_by_conf['data_type'].unique())

            num_subplots = 1
            if self.multi_subplots:
                num_data_types = len(data_types)
                num_subplots = num_data_types

            fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=((5 * num_subplots, 5)))

            if self.multi_subplots:
                for ix, data_type in enumerate(data_types):
                    data_type_info = data_by_conf[data_by_conf['data_type'] == data_type]
                    MultiEntityTypesPlotManager.plot_data(data_type_info, ax[ix], title=data_type)
            else:
                MultiEntityTypesPlotManager.plot_data(data_by_conf, ax, title=data_by_conf['scenario'].values[0])
                # MultiEntityTypesPlotManager.plot_data(data_by_conf, ax, title=data_by_conf['scenario'].values[0])

            # fig.suptitle(conf_str)
            fig.suptitle(conf_str)
            plt.show()

            self.figs.append(fig)
            self.figs_names.append("scenario={}_{}_plot.pdf".format(scenario, conf_str))

    @staticmethod
    def plot_data(data: pd.DataFrame, ax, margin_left: float = -0.05, margin_right: float = 1.05, title: str = '',
                  manage_overlap: bool = False):
        """
        This method plots the data computed by a metric in a scenario.

        :param data: Pandas DataFrame containing the data computed by a metric in a scenario.
        :param ax: axes where to plot the data
        :param margin_left: margin left of the plot
        :param margin_right: margin right of the plot
        :param title: title of the plot
        :param manage_overlap: boolean flag that indicates whether to manage overlapping points
        """
        label_map = {'perfect': '$I_P$', 'match': '$I_M$', 'concat': '$I_C$'}
        color_map = {'perfect': 'red', 'match': 'green', 'concat': 'blue'}

        for row_ix, row in data.iterrows():
            row_ix = int(row_ix)
            row = row.to_dict()
            x = row['Ix']
            y = row['Iy']

            if 'aggregation' in row:
                ax.scatter(x, y, c=color_map[row['data_type']], s=160)
                label = label_map[row['data_type']]
            else:
                ax.scatter(x, y, c=color_map[row['data_type']])
                label = row['datasets']

            if manage_overlap:
                halign = 'left'
                x_offset = 10
                if row_ix > 0:
                    halign = 'right'
                    x_offset = -10
                text = ax.annotate(label, (x, y), textcoords="offset points", xytext=(x_offset, 10),
                                   horizontalalignment=halign, verticalalignment='top')
            else:
                # if row['data_type'] == 'concat':
                #     text = ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, -25), ha='center')
                # else:
                #     text = ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 15), ha='center')
                text = ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 15), ha='center')
            text.set_fontsize(24)
            ax.tick_params(labelsize=18)

        ax.set_xlim(left=margin_left, right=margin_right)
        ax.set_ylim(bottom=margin_left, top=margin_right)

        if title:
           ax.set_title(title)

        ax.set_xlabel('source ($r_{D→I}$)', fontsize=24)
        # if title == 1:
        ax.set_ylabel('integration ($r_{I→D}$)', fontsize=24)
        ax.grid()


class MultiMatchPercentagesPlotManager(PlotManager):
    """
    This class implements a plot manager for the results produced by the MultiMatchPercentagesEvaluator.
    """

    def __init__(self, data: pd.DataFrame):
        """
        This method initializes the state of the plot manager.

        :param data: Pandas DataFrame containing the data to plot.
        """
        check_parameter_type(data, 'data', pd.DataFrame, 'Pandas DataFrame')

        super().__init__()

        self.data = data
        self.plot_dir = os.path.join(self.plot_dir, 'multi-match-percentages')

    def plot(self):
        """
        This method plots the data.
        """
        data = self.data.fillna('None')
        conf_params = ['mode', 'ngram', 'embed_manager']
        group_data_by_conf = data.groupby(conf_params)

        for conf_val, data_by_conf in group_data_by_conf:
            conf = zip(conf_params, conf_val)
            conf_str = ','.join(["{}={}".format(c[0], c[1]) for c in conf])

            # subgroups = []
            # # LOW match recall ratio -> HIGH concatenation, LOW compression
            #
            # # FIXME: discover automatically the subgroups. furthermore, note that not all the groups can be present
            # compression_subgroup = [1, 2, 3, 4, 5]
            # compression_data = data_by_conf[data_by_conf["scenario"].isin(compression_subgroup)]
            # subgroups.append(('compression', compression_data))
            #
            # concatenation_subgroup = [1, 6, 7, 8, 9]
            # concatenation_data = data_by_conf[data_by_conf["scenario"].isin(concatenation_subgroup)]
            # subgroups.append(('concatenation', concatenation_data))
            #
            # mixed_subgroup = [1, 10, 11, 12, 13]
            # mixed_data = data_by_conf[data_by_conf["scenario"].isin(mixed_subgroup)]
            # subgroups.append(('mix', mixed_data))
            #
            # for subgroup in subgroups:
            #     subgroup_name = subgroup[0]
            #     subgroup_data = subgroup[1]
            #     self.plot_subgroup_analysis(subgroup_data, subgroup_name)
            #     self.figs_names.append("{}_{}_plot.pdf".format(subgroup_name, conf_str))

            self.plot_subgroup_analysis(data_by_conf, 'mix')
            self.figs_names.append("{}_{}_plot.pdf".format('mix', conf_str))

    def plot_subgroup_analysis(self, data: pd.DataFrame, analysis_type: str):

        if analysis_type == "compression":
            labels_fun = lambda x: "{} (ACC: {:.2f})".format(x['match_ratio'], x['acc'])
        elif analysis_type == "concatenation":
            labels_fun = lambda x: "{} (ACC: {:.2f})".format(x['non_match_ratio'], x['acc'])
        elif analysis_type == "mix":
            # labels_fun = lambda x: "{}-{} ({:.2f}, {:.2f})".format(x['match_ratio'], x['non_match_ratio'], x['acc'],
            #                                                        np.linalg.norm(
            #                                                            np.array((x["Ix"], x["Iy"])) - np.array((1, 1))))
            labels_fun = lambda x: "{}-{}".format(x['match_ratio'], x['non_match_ratio'])
        else:
            raise Exception("Wrong analysis type.")

        # fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        # fig.set_size_inches((10, 5))
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        fig.set_size_inches((6, 6))
        # fig.suptitle("{} analysis".format(analysis_type))

        # # match
        # m = data[data["data_type"] == "match"]
        # # plot_data(m["Ix"].values, m["Iy"].values, m.apply(labels_fun, axis=1), ax[0], 'match', 'ob')
        #
        # # concat
        # c = data[data["data_type"] == "concat"]
        # # plot_data(c["Ix"].values, c["Iy"].values, c.apply(labels_fun, axis=1), ax[1], 'non_match', '*r')
        # MultiMatchPercentagesPlotManager.plot_data(m["Ix"].values, c["Iy"].values, c.apply(labels_fun, axis=1), ax[0],
        #                                            'match-non_match', '*r')

        # all
        a = data[data["data_type"] == "all"]
        # MultiMatchPercentagesPlotManager.plot_data(a["Ix"].values, a["Iy"].values, a.apply(labels_fun, axis=1), ax[1],
        #                                            'all', 'dg')
        MultiMatchPercentagesPlotManager.plot_data(a["Ix"].values, a["Iy"].values, a.apply(labels_fun, axis=1), ax,
                                                   '#match: {}, #non_match: {}'.format(
                                                       a['data1_data2_wrong_match'].values[0],
                                                       a['data1_data2_wrong_non_match'].values[0]), 'blue')

        plt.show()
        self.figs.append(fig)

    @staticmethod
    def plot_data(x, y, labels, ax, title, style):

        # ax.plot(x, y, style)
        ax.scatter(x, y, c=style, s=100)
        ax.set_title(title, fontdict={'fontsize': 18})
        for i, txt in enumerate(labels):
            text = ax.annotate(txt, (x[i], y[i]), ha='center', textcoords="offset points", xytext=(0, -18))
            text.set_fontsize(13)
        ax.set_xlim(0.2, 1)
        ax.set_ylim(0.2, 1)
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # ax.set_xlim(xlim[0], 1)
        # ax.set_ylim(ylim[0], 1)
        # lim = min(xlim[0], ylim[0])
        # ax.set_xlim(lim, 1)
        # ax.set_ylim(lim, 1)
        ax.set_xlabel('source ($r_{D→I}$)', fontsize=24)
        ax.set_ylabel('integration ($r_{I→D}$)', fontsize=24)
        ax.tick_params(labelsize=18)
