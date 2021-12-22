import numpy as np
import pandas as pd
import collections
from scipy.stats import entropy
import matplotlib.pyplot as plt
# from grid_strategy.strategies import SquareStrategy


class IntegrationProfiler(object):
    """
    This class manages the storage and display of the history of values recorded by a metric to profile an incremental
    integration task. This process integrates, in different iterations, an increasing number of data sources producing
    multiple integrated datasets. At each iteration multiple data sources are processed in order to generate a single
    integrated version. It is supposed to use a metric to profile the outcome of this single iteration and that this
    metric registers two set of values for each data source - integrated dataset pair (one for each direction of
    interest: from the source to the integrated or vice versa).
    """

    def __init__(self, metric_name):

        if not isinstance(metric_name, str):
            raise TypeError("Wrong data type for parameter metric_name. Only string data type is allowed")

        self.metric_name = metric_name
        self.history_scores = {}
        self.avg_history_scores = {}

    def get_history_scores(self):
        """
        This method returns the score history.
        :return: dictionary where each key represents a data integration iteration and its value the scores reported for
                 the iteration
        """
        return self.history_scores

    def get_avg_history_scores(self):
        """
        This method returns the average score history.
        :return: dictionary where each key represents a data integration iteration and its value the average scores
                 reported for the iteration
        """
        return self.avg_history_scores

    def add_integration_scores(self, idx, scores):
        """
        This method adds to the history the metric scores associated to a given iteration in the incremental
        integration task.
        :param idx: the identifier of the considered iteration of the incremental integration task
        :param scores: the registered metric scores
        :return: None
        """

        if not isinstance(idx, int):
            raise TypeError("Wrong data type for parameter idx. Only integer data type is allowed.")

        if not isinstance(scores, collections.Iterable):
            raise TypeError("Wrong data type for parameter scores. Only iterable data type is allowed.")

        for score in scores:
            if not isinstance(score, dict):
                raise TypeError("Wrong data type for scores elements. Only dictionary data type is allowed.")

        if len(list(scores)) - 1 != idx:
            raise ValueError(
                "Wrong data format for parameter scores. The length of the list has to be equal to idx + 1.")

        for score in scores:
            if "source_to_integration" not in score or "integration_to_source" not in score:
                raise ValueError(
                    "Wrong data format for parameter scores. Wrong dictionary format for the constituted elements.")

        # store iteration scores in the history
        self.history_scores[idx] = scores

        # compute and store also the score average for each data source - integration pair in the considered iteration
        avg_scores = []
        for pair_score in scores:
            avg_pair_score = {}
            avg_pair_score["source_to_integration"] = np.average(pair_score["source_to_integration"])
            avg_pair_score["integration_to_source"] = np.average(pair_score["integration_to_source"])
            avg_scores.append(avg_pair_score)
        self.avg_history_scores[idx] = avg_scores

    def get_scores_by_source(self):
        """
        This method groups history scores by source.
        :return: dictionary, where each key corresponds to a data source
        """

        scores_by_sources = {}

        # loop over the score history
        for iteration_num, iteration_scores in self.history_scores.items():

            # loop over the sources-integration pairs for the current data integration iteration
            for source_id, source_to_integration_scores in enumerate(iteration_scores):

                # get the scores registered in the direction source - integration
                source_scores = source_to_integration_scores["source_to_integration"]

                # compute the entropy
                probs_source_scores = pd.Series(source_scores).value_counts(normalize=True)
                entropy_source_scores = entropy(probs_source_scores)
                entropy_source_scores = "{:.2f}".format(entropy_source_scores)

                # save scores in output dictionary by source
                if source_id not in scores_by_sources:
                    scores_by_sources[source_id] = {
                        "Iteration #{} (H: {})".format(iteration_num, entropy_source_scores): source_scores}
                else:
                    scores_by_sources[source_id][
                        "Iteration #{} (H: {})".format(iteration_num, entropy_source_scores)] = source_scores

        return scores_by_sources

    def get_scores_by_integration(self):
        """
        This method groups history scores by integration datasets.
        :return: dictionary, where each key corresponds to an integrated dataset
        """

        scores_by_integrations = {}

        # loop over the score history
        for iteration_num, iteration_scores in self.history_scores.items():

            # loop over the sources-integration pairs for the current data integration iteration
            for source_id, source_to_integration_scores in enumerate(iteration_scores):

                # get the scores registered in the direction integration - source
                integration_scores = source_to_integration_scores["integration_to_source"]

                # compute the entropy
                probs_integration_scores = pd.Series(integration_scores).value_counts(normalize=True)
                entropy_integration_scores = entropy(probs_integration_scores)
                entropy_integration_scores = "{:.2f}".format(entropy_integration_scores)

                # save scores in output dictionary by integration
                if source_id not in scores_by_integrations:
                    scores_by_integrations[iteration_num] = {
                        "Source #{} (H: {})".format(source_id, entropy_integration_scores): integration_scores}
                else:
                    scores_by_integrations[iteration_num][
                        "Source #{} (H: {})".format(source_id, entropy_integration_scores)] = integration_scores

        return scores_by_integrations

    def get_aggregated_average_scores_by_iteration(self):
        """
        This method aggregates with the mean and flattens (for a given data integration iteration) the average scores
        between each data source - integration pairs.
        :return: (means for direction source to integration, means for direction integration to source)
        """

        s_ax = []
        i_ax = []
        for iteration in self.avg_history_scores:
            iteration_scores = self.avg_history_scores[iteration]
            # compute for each iteration the mean of the BM25 values in both directions
            # FIXME: ho aggiunto la selezione dei valori unici perché non volevo che multiple coppie sorgente-integrata
            # FIXME: con lo stesso score influenzasse il calcolo
            # s_ax.append(np.mean(np.unique([pair_scores["source_to_integration"] for pair_scores in iteration_scores])))
            # i_ax.append(np.mean(np.unique([pair_scores["integration_to_source"] for pair_scores in iteration_scores])))
            s_ax.append(np.mean([pair_scores["source_to_integration"] for pair_scores in iteration_scores]))
            i_ax.append(np.mean([pair_scores["integration_to_source"] for pair_scores in iteration_scores]))

        return s_ax, i_ax

    def get_average_scores_grouped_by_iteration(self):
        """
        This method groups average historical scores by iteration.
        :return: list of lists of average scores for each iteration
        """

        points_by_iteration = []

        for iteration in self.avg_history_scores:
            iteration_scores = self.avg_history_scores[iteration]
            s_ax = [pair_scores["source_to_integration"] for pair_scores in iteration_scores]
            i_ax = [pair_scores["integration_to_source"] for pair_scores in iteration_scores]

            points_by_iteration.append([s_ax, i_ax])

        return points_by_iteration

    # def plot_histograms_by_sources(self):
    #     """
    #     This method plots the history scores by source with histograms.
    #     :return: None
    #     """
    #
    #     # get history scores grouped by data source
    #     scores_by_sources = self.get_scores_by_source()
    #
    #     # define the grid plotting strategy
    #     grid_strategy = SquareStrategy()
    #     grid_dims = grid_strategy.get_grid_arrangement(n=len(scores_by_sources))
    #     nrows = len(grid_dims)
    #     if nrows == 0:
    #         nrows = 1
    #     ncols = np.max(grid_dims)
    #
    #     plt.figure()
    #
    #     # plot histograms for each data source
    #     for source_id in scores_by_sources:
    #         my_ax = plt.subplot(nrows, ncols, source_id + 1)
    #         my_ax.set_title("Source {}".format(source_id))
    #         my_ax.set_xlabel("{} scores".format(self.metric_name))
    #         source_data = pd.DataFrame(scores_by_sources[source_id])
    #         source_data.plot.hist(ax=my_ax, bins=10, alpha=0.5)
    #         my_ax.set_ylabel("")
    #         my_ax.set_xlim(0)
    #
    #     plt.show()
    #
    # def plot_histograms_by_integrations(self):
    #     """
    #     This method plots the history scores by integration with histograms.
    #     :return: None
    #     """
    #
    #     # get history scores grouped by integration datasets
    #     scores_by_integrations = self.get_scores_by_integration()
    #
    #     # define the grid plotting strategy
    #     grid_strategy = SquareStrategy()
    #     grid_dims = grid_strategy.get_grid_arrangement(n=len(scores_by_integrations))
    #     nrows = len(grid_dims)
    #     if nrows == 0:
    #         nrows = 1
    #     ncols = np.max(grid_dims)
    #
    #     plt.figure()
    #
    #     # plot histograms for each integrated dataset
    #     for integration_id in scores_by_integrations:
    #         my_ax = plt.subplot(nrows, ncols, integration_id)
    #         my_ax.set_title("Integration {}".format(integration_id))
    #         my_ax.set_xlabel("{} scores".format(self.metric_name))
    #
    #         # force all the arrays to have the same length by adding padding elements (i.e., np.nan)
    #         scores_same_length = {}
    #         max_length = np.max([len(array) for array in scores_by_integrations[integration_id].values()])
    #         for source_id, integration_scores in scores_by_integrations[integration_id].items():
    #             len_itegration_scores = len(integration_scores)
    #             if len_itegration_scores < max_length:
    #                 integration_scores = integration_scores + [np.nan] * (max_length - len_itegration_scores)
    #             scores_same_length[source_id] = integration_scores
    #
    #         source_data = pd.DataFrame(scores_same_length)
    #         source_data.plot.hist(ax=my_ax, bins=10, alpha=0.5)
    #         my_ax.set_ylabel("")
    #         my_ax.set_xlim(0)
    #
    #     plt.show()

    def plot_average_history_scores(self, show=True):
        """
        This method plots history average scores in a single line plot.
        :param show: boolean flag for enabling the plot show
        :return: None
        """

        # get history average scores by iteration
        x, y = self.get_aggregated_average_scores_by_iteration()

        # plot line with marckers
        plt.plot(x, y, marker='o', markersize=8, linewidth=1)

        # add annotations to markers
        index = 1
        for single_x, single_y in zip(x, y):
            label = "{}".format(index)
            index += 1

            plt.annotate(label,
                         (single_x, single_y),
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

        plt.xlabel("source ({})".format(self.metric_name))
        plt.ylabel("integration ({})".format(self.metric_name))
        plt.axis('square')
        plt.grid(True)
        if self.metric_name == "Jaccard":
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
        elif self.metric_name == "BM25":
            plt.xlim(2, 6)
            plt.ylim(2, 6)
            # pass
        # fig = plt.gcf()
        # fig.set_size_inches(5, 5)

        if show:
            plt.show()

    def plot_iteration_average_and_single_history_scores(self, single_scores, matching_iteration, average_scores,
                                                         new_figure=True, color='r'):
        """
        This method plots (for a given iteration) the raw and average historical scores in multiple plots.

        :param single_scores: raw historical scores for the considered iteration
        :param matching_iteration: the integration iteration
        :param average_scores: average historical scored for the considered iteration
        :param new_figure: boolean flag to indicate whether plot on a new figure
        :param color: string indicating the color for score points
        :return: None
        """

        x = average_scores[0]  # direction source to integration
        y = average_scores[1]  # direction integration to source

        if new_figure:
            plt.figure()

        # plot raw scores as single points
        plt.plot(single_scores[0], single_scores[1], marker='s', c=color, linestyle='None', markersize=8)

        # add annotations for the raw score points
        source_id = 1
        for single_x, single_y in zip(single_scores[0], single_scores[1]):
            label = "S{} ({})".format(source_id, matching_iteration + 1)
            source_id += 1
            plt.annotate(label,
                         (single_x, single_y),
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

        # plot average scores with a line
        plt.plot(x[:matching_iteration + 1], y[:matching_iteration + 1],
                 marker='o', markersize=8, linewidth=1)

        # add annotation for average scores
        index = 1
        for cum_single_x, cum_single_y in zip(x[:matching_iteration + 1], y[:matching_iteration + 1]):
            label = "{}".format(index)
            index += 1

            plt.annotate(label,
                         (cum_single_x, cum_single_y),
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

        plt.xlabel("source ({})".format(self.metric_name))
        plt.ylabel("integration ({})".format(self.metric_name))

        plt.grid(True)
        plt.axis('square')
        if self.metric_name == "Jaccard":
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
        elif self.metric_name == "BM25":
            plt.xlim(2, 6)
            plt.ylim(2, 6)
            # pass

        if new_figure:
            plt.show()

    def plot_average_and_single_history_scores(self):
        """
        This method plots in different plots the raw and aggregated historical average scores.
        :return: None
        """

        # get average scores aggregated by mean and grouped by iteration
        x, y = self.get_aggregated_average_scores_by_iteration()
        avg_scores = [x, y]

        # get average scores grouped by iteration
        points_by_iteration = self.get_average_scores_grouped_by_iteration()

        # plot each iteration scores in a different figure
        for matching_iteration, single_scores in enumerate(points_by_iteration):
            self.plot_iteration_average_and_single_history_scores(single_scores, matching_iteration, avg_scores)


class CompareIntegrationProfiler(object):
    """
    This class manages the comparison between two or more IntegrationProfilers.
    """

    def __init__(self, profilers, labels):
        """
        This method saves the profilers to compare.
        :param profilers: list of IntegrationProfiler objects to compare
        :param labels: list of labels associated to the user-provided profilers
        """

        if not isinstance(profilers, collections.Iterable):
            raise TypeError("Wrong data type for parameter profilers. Only iterable data type is allowed.")

        if not isinstance(labels, collections.Iterable):
            raise TypeError("Wrong data type for parameter labels. Only iterable data type is allowed.")

        for profiler in profilers:
            if not isinstance(profiler, IntegrationProfiler):
                raise TypeError(
                    "Wrong data type for profilers elements. Only IntegrationProfiler data type is allowed.")

        for label in labels:
            if not isinstance(label, str):
                raise TypeError("Wrong data type for labels elements. Only string data type is allowed.")

        if len(list(profilers)) < 1:
            raise ValueError("Wrong value for parameter profilers. The length has to be greater than 1.")

        if len(list(labels)) < 1:
            raise ValueError("Wrong value for parameter labels. The length has to be greater than 1.")

        if len(list(profilers)) != len(list(labels)):
            raise ValueError("Lengths of parameters profilers and labels don't match.")

        self.profilers = list(profilers)
        self.labels = list(labels)
        self.colors = ["g", "r", "c", "m", "y"]

    def plot_average_history_scores(self, subplots=False):
        """
        This method plots in a single chart the average history scores of multiple profilers.
        :param subplots: boolean flag that indicates whether to plot profiler scores in the same or different subplots
        :return: None
        """

        # loop over profilers
        for profiler_id, profiler in enumerate(self.profilers):

            if subplots:
                plt.subplot(1, len(self.profilers), profiler_id + 1)

            # plot the average history scores for one profiler at a time
            profiler.plot_average_history_scores(show=False)

            if subplots:
                plt.title(self.labels[profiler_id])

        if not subplots:
            plt.legend(self.labels)

        plt.show()

    def plot_average_and_single_history_scores(self, subplots=False):
        """
        This method plots in a single chart (one for each integration iteration) the aggregated and single history
        scores of multiple profilers.
        :param subplots: boolean flag that indicates whether to plot profiler scores (for a specific integration
               iteration) in the same or different subplots
        :return: None
        """

        profiler_aggregated_scores = []
        profiler_scores = []

        # loop over profilers
        for profiler in self.profilers:
            # get average scores aggregated by mean and grouped by iteration
            x, y = profiler.get_aggregated_average_scores_by_iteration()
            avg_scores = [x, y]
            profiler_aggregated_scores.append(avg_scores)

            # get average scores grouped by iteration
            points_by_iteration = profiler.get_average_scores_grouped_by_iteration()
            profiler_scores.append(points_by_iteration)

        num_iterations = len(profiler_scores[0])

        # loop over integration iterations
        for matching_iteration in range(num_iterations):

            # loop over profilers
            for profiler_id, profiler in enumerate(self.profilers):
                agg_scores = profiler_aggregated_scores[profiler_id]
                scores = profiler_scores[profiler_id][matching_iteration]

                if subplots:
                    plt.subplot(1, len(self.profilers), profiler_id + 1)

                # draw in the same plot the scores of the profilers at the same integration iteration
                profiler.plot_iteration_average_and_single_history_scores(scores, matching_iteration, agg_scores,
                                                                          new_figure=False,
                                                                          color=self.colors[profiler_id])

                if subplots:
                    plt.title(self.labels[profiler_id])

            if not subplots:
                new_labels = []
                for label in self.labels:
                    new_labels.append(label)
                    new_labels.append("{} (agg)".format(label))
                plt.legend(new_labels)

            plt.show()
