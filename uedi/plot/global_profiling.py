import pandas as pd
import matplotlib.pyplot as plt

from uedi.nlp.language_models import LanguageModel, get_distribution


def plot_distribution(source, integration, stacked=False):
    docs_integration = [x.split() for x in integration]
    lm_integration = LanguageModel(n=1, model='mle')
    lm_integration.fit(docs_integration)
    vocabs, scores = lm_integration.get_distribution()
    s_integration = pd.Series(scores, index=vocabs)

    docs_source = [x.split() for x in source]
    lm_source = LanguageModel(n=1, model='mle')
    lm_source.fit(docs_source)
    vocabs, scores = lm_source.get_distribution()
    s_source = pd.Series(scores, index=vocabs)

    ds = pd.concat([s_integration, s_source], axis=1)
    ds.fillna(0, inplace=True)
    ds.columns = ['integration', 'source']

    if stacked:
        fig, axes = plt.subplots(figsize=(14, 7))
        ds.plot.bar(ax=axes, stacked=False)
        plt.show()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True, sharey=True)
        ds['source'].plot.bar(ax=axes[0], color='blue', title='source')
        ds['integration'].plot.bar(ax=axes[1], color='orange', title='integration')
        plt.show()


def plot_intersection(source, integration):
    source = get_distribution(source)
    integration = get_distribution(integration)

    ds = pd.concat([source, integration], axis=1)
    ds.fillna(0, inplace=True)
    ds.columns = ['source', 'integration']

    fig, axes = plt.subplots(figsize=((14, 7)))
    ds[ds['source'] != 0].plot.bar(ax=axes, stacked=False)
    plt.show()


def plot_two_res(res_scores_match, res_scores_concat,
                 resize=False,
                 left=-0.05, right=1,
                 title='', title_a='', title_b=''):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=((10, 5)))

    max_x = 0
    max_y = 0
    for i, d in res_scores_match.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):

            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            print("\tcoord({}-{}) = ({:.2f}, {:.2f})".format(i, j, x, y))

            axes[0].scatter(x, y, c='green')
            axes[0].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    if max_x <= right and max_y <= right:
        axes[0].set_xlim(left=left, right=right)
        axes[0].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[0].set_xlim(left=left, right=max_x)
        axes[0].set_ylim(bottom=left, top=max_y)

    if title_a:
        axes[0].set_title(title_a)
    else:
        axes[0].set_title('with match')

    axes[0].set_xlabel('source')
    axes[0].set_ylabel('integration')
    axes[0].grid()

    for i, d in res_scores_concat.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):
            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            print("\tcoord({}-{}) = ({:.2f}, {:.2f})".format(i, j, x, y))

            axes[1].scatter(x, y, c='green')
            axes[1].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    if max_x <= right and max_y <= right:
        axes[1].set_xlim(left=left, right=right)
        axes[1].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[1].set_xlim(left=left, right=max_x)
        axes[1].set_ylim(bottom=left, top=max_y)

    if title_b:
        axes[1].set_title(title_b)
    else:
        axes[1].set_title('with concat')

    axes[1].set_xlabel('source')
    axes[1].set_ylabel('integration')
    axes[1].grid()

    fig.suptitle(title, fontsize=16)
    plt.show()






def plot_two_res_old(res_scores_match, res_scores_concat,
                 resize=False,
                 left=-0.05, right=1,
                 title=''):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))

    max_x = 0
    max_y = 0
    print("Sub-scenario A")
    for i, d in res_scores_match.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):

            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            print("\tcoord({}-{}) = ({:.2f}, {:.2f})".format(i, j, x, y))
            axes[0].scatter(x, y, c='green')
            axes[0].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    if max_x <= right and max_y <= right:
        axes[0].set_xlim(left=left, right=right)
        axes[0].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[0].set_xlim(left=left, right=max_x)
        axes[0].set_ylim(bottom=left, top=max_y)

    # axes[0].set_title('with match')
    axes[0].set_title('Sub-scenario A')
    axes[0].set_xlabel('source')
    axes[0].set_ylabel('integration')
    axes[0].grid()

    print("Sub-scenario B")
    for i, d in res_scores_concat.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):
            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            print("\tcoord({}-{}) = ({:.2f}, {:.2f})".format(i, j, x, y))
            axes[1].scatter(x, y, c='green')
            axes[1].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    if max_x <= right and max_y <= right:
        axes[1].set_xlim(left=left, right=right)
        axes[1].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[1].set_xlim(left=left, right=max_x)
        axes[1].set_ylim(bottom=left, top=max_y)

    # axes[1].set_title('with concat')
    axes[1].set_title('Sub-scenario B')
    axes[1].set_xlabel('source')
    axes[1].set_ylabel('integration')
    axes[1].grid()

    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_two_res_with_user_data(res_scores_match, res_scores_concat, user_data,
                 resize=False,
                 left=-0.05, right=1,
                 title=''):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))

    max_x = 0
    max_y = 0
    for i, d in res_scores_match.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):

            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            axes[0].scatter(x, y, c='green')
            axes[0].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    # plot user data
    caseA_data = user_data['caseA']
    idx = 0
    for x, y in zip(caseA_data['x'], caseA_data['y']):
        full_label = caseA_data["labels"][idx]
        label_items = full_label.split()
        label = label_items[0]
        metric = label_items[1]
        color = None
        if metric == "F1":
            color = "blue"
        elif metric == "F1B":
            color = "red"
        axes[0].scatter(x, y, c=color)
        axes[0].annotate(label, (x, y),
                         textcoords="offset points", xytext=(0, -10), ha='center')

        idx += 1

    if max_x <= right and max_y <= right:
        axes[0].set_xlim(left=left, right=right)
        axes[0].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[0].set_xlim(left=left, right=max_x)
        axes[0].set_ylim(bottom=left, top=max_y)

    # axes[0].set_title('with match')
    axes[0].set_title('Sub-scenario A')
    axes[0].set_xlabel('source')
    axes[0].set_ylabel('integration')
    axes[0].grid()

    for i, d in res_scores_concat.items():
        j = 0
        for x, y in zip(d['source'], d['integration']):
            if resize and x > max_x:
                max_x = x
            if resize and y > max_y:
                max_y = y

            axes[1].scatter(x, y, c='green')
            axes[1].annotate('{}-{}'.format(i, j), (x, y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
            j += 1

    # plot user data
    caseB_data = user_data['caseB']
    idx = 0
    for x, y in zip(caseB_data['x'], caseB_data['y']):
        full_label = caseB_data["labels"][idx]
        label_items = full_label.split()
        label = label_items[0]
        metric = label_items[1]
        color = None
        if metric == "F1":
            color = "blue"
        elif metric == "F1B":
            color = "red"
        axes[1].scatter(x, y, c=color)
        axes[1].annotate(label, (x, y),
                         textcoords="offset points", xytext=(0, -10), ha='center')

        idx += 1

    if max_x <= right and max_y <= right:
        axes[1].set_xlim(left=left, right=right)
        axes[1].set_ylim(bottom=left, top=right)
    else:
        if max_y < right:
            max_y = right
        if max_x < right:
            max_x = right
        axes[1].set_xlim(left=left, right=max_x)
        axes[1].set_ylim(bottom=left, top=max_y)

    # axes[1].set_title('with concat')
    axes[1].set_title('Sub-scenario B')
    axes[1].set_xlabel('source')
    axes[1].set_ylabel('integration')
    axes[1].grid()

    fig.suptitle(title, fontsize=16)
    plt.show()