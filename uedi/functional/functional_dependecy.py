import numpy as np
import matplotlib.pyplot as plt


class FunctionalDependency:

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.history_prc = {}
        self.history_acc = {}

    def get(self):
        return [self.lhs, self.rhs]

    def get_acc_points(self):
        ax_i = [x['ti'] / (x['ti'] + x['fi']) if x['ti'] != 0 or x['fi'] != 0 else 0 for x in self.history_acc.values()]
        ax_s = [x['ts'] / (x['ts'] + x['fs']) if x['ts'] != 0 or x['fs'] != 0 else 0 for x in self.history_acc.values()]
        return ax_i, ax_s

    def get_acc_cum_points(self):
        ax_i, ax_s = self.get_acc_points()
        ax_i = np.cumsum(ax_i)
        ax_s = np.cumsum(ax_s)

        ax_i = ax_i / list(range(1, len(ax_i) + 1))
        ax_s = ax_s / list(range(1, len(ax_s) + 1))

        return ax_i, ax_s

    def get_micro(self):
        ax_i = []
        ax_s = []

        for i, matches in enumerate(self.history_prc.values()):
            ts = 0
            fs = 0
            ti = 0
            fi = 0
            for x in matches:
                ts += x['ts']
                fs += x['fs']
                ti += x['ti']
                fi += x['fi']

            ax_s.append(ts / (ts + fs) if ts != 0 or fs != 0 else 0)
            ax_i.append(ti / (ti + fi) if ti != 0 or fi != 0 else 0)

        return ax_i, ax_s

    # def get_macro_points(self):
    #     ax_i = []
    #     ax_s = []
    #
    #     for matches in self.history_prc.values():
    #         ax_i += [x['ti'] / (x['ti'] + x['fi']) if x['ti'] != 0 or x['fi'] != 0 else 0 for x in matches]
    #         ax_s += [x['ts'] / (x['ts'] + x['fs']) if x['ts'] != 0 or x['fs'] != 0 else 0 for x in matches]
    #
    #     return ax_i, ax_s

    def get_macro(self):
        ax_i = []
        ax_s = []

        for i, matches in enumerate(self.history_prc.values()):
            # compute point i for axis source
            ms = np.mean([x['ts'] / (x['ts'] + x['fs']) if x['ts'] != 0 or x['fs'] != 0 else 0 for x in matches])
            ax_s.append(ms)

            # compute point i for axis integration
            tot_tuple_source = np.sum([x['ts'] + x['fs'] for x in matches])
            tot_tuple_integration = matches[0]['ti'] + matches[0]['fi']

            mi = np.sum([x['ti'] / (x['ti'] + x['fi']) if x['ti'] != 0 or x['fi'] != 0 else 0 for x in matches])
            mi = mi * (tot_tuple_integration / tot_tuple_source)

            ax_i.append(mi)

        return ax_i, ax_s

    def update(self, idx, ti, fi, ts, fs):
        self.history_prc[idx] = {}
        self.history_prc[idx]['ti'] = ti
        self.history_prc[idx]['fi'] = fi
        self.history_prc[idx]['ts'] = ts
        self.history_prc[idx]['fs'] = fs

    def update_acc(self, idx, ti, fi, ts, fs):
        self.history_acc[idx] = {}
        self.history_acc[idx]['ti'] = ti
        self.history_acc[idx]['fi'] = fi
        self.history_acc[idx]['ts'] = ts
        self.history_acc[idx]['fs'] = fs

    def update_with_history(self, idx, matches):
        self.history_prc[idx] = matches

    # def get_macro_points_by_source(self):
    #
    #     points_by_source = [[[], []] for _ in range(len(self.history_prc.values()))]
    #
    #     for matches in self.history_prc.values():
    #         for j, match in enumerate(matches):
    #             score_i = 0
    #             if match['ti'] != 0 or match['fi'] != 0:
    #                 score_i = match['ti'] / (match['ti'] + match['fi'])
    #
    #             score_s = 0
    #             if match['ts'] != 0 or match['fs'] != 0:
    #                 score_s = match['ts'] / (match['ts'] + match['fs'])
    #
    #             points_by_source[j][0].append(score_s)
    #             points_by_source[j][1].append(score_i)
    #
    #     print(points_by_source)
    #
    #     return points_by_source

    def get_macro_points_by_iteration(self):
        """
        This function groups macro precision scores by data integration iteration.
        The adopted output data format is the following:
        [macro_scores_iteration_1, macro_scores_iteration2, ...]
          - macro_scores_iteration1: [macro_precision_scores_x, macro_precision_scores_y]
            - macro_precision_scores_x: [macro_precision_source1_integration_x, macro_precision_source2_integration_x]
            - macro_precision_scores_y: [macro_precision_source1_integration_y, macro_precision_source2_integration_y]
        :return:
        """

        points_by_iteration = []

        for matches in self.history_prc.values():
            ax_i = [x['ti'] / (x['ti'] + x['fi']) if x['ti'] != 0 or x['fi'] != 0 else 0 for x in matches]
            ax_s = [x['ts'] / (x['ts'] + x['fs']) if x['ts'] != 0 or x['fs'] != 0 else 0 for x in matches]

            points_by_iteration.append([ax_s, ax_i])

        return points_by_iteration

    def _plot_macro_precisions_and_centroid(self, macro_precisions, matching_iteration, macro_precisions_centroid):

        x = macro_precisions_centroid[0]
        y = macro_precisions_centroid[1]

        plt.figure()

        # plot the macro scores of the last data integration iteration only
        plt.plot(macro_precisions[0], macro_precisions[1],
                 marker='s', c='r', linestyle='None',
                 markersize=8)

        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)

        # add annotations for macro precision scores
        source_id = 1
        for single_x, single_y in zip(macro_precisions[0], macro_precisions[1]):
            label = "S{} ({})".format(source_id, matching_iteration + 1)
            source_id += 1
            plt.annotate(label,
                         (single_x, single_y),
                         textcoords="offset points",    # how to position the text
                         xytext=(0, 10),                # distance from text to points (x,y)
                         ha='center')                   # horizontal alignment can be left, right or center

        # plot centroid of macro precision scores
        plt.plot(x[:matching_iteration + 1], y[:matching_iteration + 1],
                 marker='o', markersize=8, linewidth=1)

        # add annotation for centroid of macro precision scores
        index = 1
        for cum_single_x, cum_single_y in zip(x[:matching_iteration + 1], y[:matching_iteration + 1]):
            label = "{}".format(index)
            index += 1

            plt.annotate(label,
                         (cum_single_x, cum_single_y),
                         textcoords="offset points",    # how to position the text
                         xytext=(0, 10),                # distance from text to points (x,y)
                         ha='center')                   # horizontal alignment can be left, right or center

        plt.xlabel("source (precision)")
        plt.ylabel("integration (precision)")

        plt.title("macro prc iteration {}  FD: {} -> {}"
                  .format(matching_iteration + 1, self.lhs, self.rhs), loc='center', pad=20)
        plt.grid(True)
        plt.axis('square')
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.show()
        # fig = plt.gcf()
        # fig.set_size_inches(5, 5)
        # fig.savefig("figures/scenario1_macro{}.png".format(iteration))

    def plot_macro(self, single_point=True):

        # get centroid of macro precision scores
        y, x = self.get_macro()
        macro_precisions_centroid = [x, y]

        # get macro precision scores grouped by data integration iteration
        points_by_iteration = self.get_macro_points_by_iteration()

        if single_point:

            macro_precisions = points_by_iteration[-1]
            matching_iteration = len(points_by_iteration)
            # plot the macro precision scores and the related centroid of the last data integration iteration only
            self._plot_macro_precisions_and_centroid(macro_precisions, matching_iteration, macro_precisions_centroid)

        else:

            # loop over data integration iteration macro precision scores
            for matching_iteration, macro_precisions in enumerate(points_by_iteration):

                # plot macro precision scores and related centroid
                self._plot_macro_precisions_and_centroid(macro_precisions, matching_iteration,
                                                         macro_precisions_centroid)

    def plot_micro(self):

        # get micro precision scores
        y, x = self.get_micro()

        # plot micro precision scores
        plt.plot(x, y, marker='o', c='b', markersize=8, linewidth=1)

        # add annotations to micro precision scores
        index = 1
        for single_x, single_y in zip(x, y):
            label = "{}".format(index)
            index += 1

            plt.annotate(label,
                         (single_x, single_y),
                         textcoords="offset points",    # how to position the text
                         xytext=(0, 10),                # distance from text to points (x,y)
                         ha='center')                   # horizontal alignment can be left, right or center

        plt.xlabel("source (precision)")
        plt.ylabel("integration (precision)")
        plt.title("micro prc FD: {} -> {}".format(self.lhs, self.rhs), loc='center', pad=20)
        plt.axis('square')
        plt.grid(True)
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        # fig = plt.gcf()
        # fig.set_size_inches(5, 5)
        plt.show()

    def plot_acc(self):
        y, x = self.get_acc_points()
        # y_cum, x_cum = self.get_acc_cum_points()

        plt.plot(x, y, marker='o', markersize=8, linewidth=1)

        index = 1
        for x, y in zip(x, y):
            label = "{}".format(index)
            index += 1

            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

        # plt.scatter(x_cum, y_cum, s=40, c='r', marker='+')
        plt.xlabel("source")
        plt.ylabel("integration")
        plt.title("acc FD: {} -> {}".format(self.lhs, self.rhs), loc='center', pad=20)
        plt.axis('square')
        plt.grid(True)
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.show()

    def __eq__(self, other):
        if isinstance(other, FunctionalDependency):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __str__(self):
        s = "lhs: {}\n".format(self.lhs)
        s += "rhs: {}\n".format(self.rhs)
        for v, k in self.history_prc.items():
            s += "{}\n".format(v)
            s += "\t{}\n".format(k)
        return s


