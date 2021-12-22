from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import MLE, Laplace, Lidstone, WittenBellInterpolated, InterpolatedLanguageModel
import pandas as pd
import numpy as np
from scipy.stats import entropy
from uedi.nlp.pre_processing import TextPreparation
import collections
from scipy.interpolate import interp1d


class LanguageModelProfiler(object):
    """
    This class implements a series of dataset-level metrics based on language models.
    """

    @staticmethod
    def kl_language_model(lm1, lm2):
        """
        This method compares two language models by measuring their distance in terms of KL divergence. This is a non-
        symmetric measure as the two word distributions are compared over the vocabs of the first language model.

        :param lm1: language model 1
        :param lm2: language model 2
        :return: KL divergence between the two language models
        """

        if not isinstance(lm1, LanguageModel):
            raise TypeError("Wrong data type for parameter lm1. Only LanguageModel data type is allowed.")

        if not isinstance(lm2, LanguageModel):
            raise TypeError("Wrong data type for parameter lm2. Only LanguageModel data type is allowed.")

        vocabs1, scores1 = lm1.get_probs()

        a = np.asarray(scores1, dtype=np.float)

        b = [lm2.score_word(w) for w in vocabs1]
        b = np.asarray(b, dtype=np.float)

        return entropy(a, b)

    @staticmethod
    def kl_smoothed(lm1, lm2, lm_collection=None, lambd=0.1):
        """
        This method compares two language models by measuring their distance in terms of KL divergence. This is a non-
        symmetric measure as the two word distributions are compared over the vocabs of the first language model.
        A smoothed function can be applied to manage the unseen words (i.e., the words of the first language model that
        don't appear in the second one).

        :param lm1: language model 1
        :param lm2: language model 2
        :param lm_collection: union of the language models
        :param lambd: weight in the smoothing function
        :return: KL divergence between the two (smoothed) language models
        """

        if not isinstance(lm1, LanguageModel):
            raise TypeError("Wrong data type for parameter lm1. Only LanguageModel data type is allowed.")

        if not isinstance(lm2, LanguageModel):
            raise TypeError("Wrong data type for parameter lm2. Only LanguageModel data type is allowed.")

        if lm_collection:
            if not isinstance(lm_collection, LanguageModel):
                raise TypeError("Wrong data type for parameter lm_collection. Only LanguageModel data type is allowed.")

        if not isinstance(lambd, float):
            raise TypeError("Wrong data type for parameter lambd. Only float data type is allowed.")

        if lambd < 0 or lambd > 1:
            raise TypeError("Wrong value for parameter lambd. Only values in the range [0, 1] are allowed.")

        vocabs1, scores1 = lm1.get_probs()

        # if lm_collection:
        #     scores1 = [lambd * score + (1 - lambd) * lm_collection.score_word(vocab) for vocab, score in
        #                zip(vocabs1, scores1)]

        a = np.asarray(scores1, dtype=np.float)

        # print("Sum ML1: {}".format(np.sum(a)))
        eps = 0.000001
        assert 1 + eps > np.sum(a) > 1 - eps

        if lm_collection:
            b = [lambd * lm2.score_word(vocab) + (1 - lambd) * lm_collection.score_word(vocab) for vocab in vocabs1]
        else:
            b = [lm2.score_word(w) for w in vocabs1]

        # normalization
        b = [item / np.sum(b) for item in b]

        b = np.asarray(b, dtype=np.float)

        # print("Sum ML2: {}".format(np.sum(b)))
        assert 1 + eps > np.sum(b) > 1 - eps

        return entropy(a, b)

    @staticmethod
    def lm_overlap(lm1, lm2, debug=False):
        """
        This method compares two language models by measuring their distance in terms of extent of tokens in the common
        vocabulary. This is a non-symmetric measure as to the token distribution of the first language model built over
        the common vocabulary is subtracted the minimum token distribution over the common vocabulary.

        :param lm1: language model 1
        :param lm2: language model 2
        :param debug: flag for enabling the debug modality
        :return: distance between language models in terms of extent of token distribution over the common vocabulary
        """

        if not isinstance(lm1, LanguageModel):
            raise TypeError("Wrong data type for parameter lm1. Only LanguageModel data type is allowed.")

        if not isinstance(lm2, LanguageModel):
            raise TypeError("Wrong data type for parameter lm2. Only LanguageModel data type is allowed.")

        vocabs1 = lm1.get_vocabs()
        vocabs2 = lm2.get_vocabs()

        if debug:
            print(vocabs1)
            print(vocabs2)

        common_vocabs = set(vocabs1).intersection(vocabs2)
        common_vocabs = common_vocabs.difference({'<UNK>'})

        if debug:
            print(common_vocabs)

        if len(common_vocabs) == 0:
            if debug:
                print("NULL")
            return 0

        common_vocabs_scores1 = [lm1.model.counts[word] for word in common_vocabs]
        common_vocabs_scores2 = [lm2.model.counts[word] for word in common_vocabs]

        min_common_freq = np.minimum(common_vocabs_scores1, common_vocabs_scores2)

        if debug:
            print("NOT NULL")
            print(common_vocabs_scores1)
            print(np.sum(common_vocabs_scores1))
            print(min_common_freq)
            print(np.sum(min_common_freq))

        return np.sum(common_vocabs_scores1) - np.sum(min_common_freq)

    @staticmethod
    def compare_text_language_models(text1, text2, metric, topk=1, debug=False):
        """
        This method compares the similarity of two texts by measuring the distance of the language models built over
        each pair of sentences from text1 to text2. The distance can be measured with multiple methods: KL divergence
        or evaluating the overlap between language models.

        :param text1: text1
        :param text2: text2
        :param metric: the name of the metric to be used to evaluate the distance between language models
        :param topk: the optional number of top-k sentences to be considered to associate to each sentence a single
                     similarity score
        :param debug: flag for enabling the debug modality
        :return: list containing the similarity scores between each record of the first dataframe with respect the
                 second one
        """

        if not isinstance(text1, collections.Iterable):
            raise TypeError("Wrong data type for parameter text1. Only iterable data type is allowed.")

        for word_tokens1 in text1:
            if not isinstance(word_tokens1, collections.Iterable):
                raise TypeError("Wrong data type for word tokens of text1. Only iterable data type is allowed.")

        for word_tokens1 in text1:
            for token1 in word_tokens1:
                if not isinstance(token1, str):
                    raise TypeError("Wrong data type for tokens of text1. Only string data type is allowed.")

        if not isinstance(text2, collections.Iterable):
            raise TypeError("Wrong data type for parameter text2. Only iterable data type is allowed.")

        for word_tokens2 in text2:
            if not isinstance(word_tokens2, collections.Iterable):
                raise TypeError("Wrong data type for word tokens of text2. Only iterable data type is allowed.")

        for word_tokens2 in text2:
            for token2 in word_tokens2:
                if not isinstance(token2, str):
                    raise TypeError("Wrong data type for tokens of text2. Only string data type is allowed.")

        if not isinstance(metric, str):
            raise TypeError("Wrong data type for parameter metric. Only string data type is allowed.")

        if not isinstance(topk, int):
            raise TypeError("Wrong data type for parameter topk. Only integer data type is allowed.")

        if topk <= 0 or topk > len(text2):
            raise TypeError(
                "Wrong data value for parameter topk. Only values in the range [1, {}] are allowed.".format(
                    len(text2)))

        metrics = ["kl", "overlap"]
        if metric not in metrics:
            raise ValueError("Wrong data value for parameter metric. Only values {} are allowed.".format(metrics))

        scores = []

        # loop over the sentences of the first text
        for sentence1_id, sentence1 in enumerate(text1):

            # build a language model over the current sentence of the first text
            lm1 = LanguageModel(n=1, mtype='mle')
            # lm1 = LanguageModel(n=1, mtype='laplace')
            lm1.fit([sentence1])

            if debug:
                print("\tSENTENCE {} (from text1): {}".format(sentence1_id, sentence1))

            sentence_scores = []

            # loop over the sentences of the second text
            for sentence2_id, sentence2 in enumerate(text2):
                # print(sentence2_id)
                # if sentence2_id == 12:
                #     exit(1)

                # build a language model over the current sentence of the second text
                lm2 = LanguageModel(n=1, mtype='mle')
                # lm2 = LanguageModel(n=1, mtype='laplace')
                lm2.fit([sentence2])

                score = None
                if metric == "kl":

                    lm1_and_lm2 = LanguageModel(n=1, mtype='mle')
                    lm1_and_lm2.fit([sentence1, sentence2])

                    # compute KL divergence between the two language models
                    score = LanguageModelProfiler.kl_smoothed(lm1, lm2, lm_collection=lm1_and_lm2)
                    # score = LanguageModelProfiler.kl_language_model(lm1, lm2)

                    # FIXME: accrocchio per sistemare la presenza di score negativi che si generano quando i due language
                    # FIXME: non hanno elementi in comune
                    if score < 0:
                        score = (score * - 1) + 0.1

                    # FIXME: accrocchio per sistemare la presenza di score a 0 anche se i language model non coincidono in
                    # FIXME: maniera esatta. Sfrutto la lunghezza delle frasi per discriminare questi due casi
                    if score == 0:
                        diff_len = abs(len(sentence1) - len(sentence2))
                        score += diff_len * 0.0001

                    # convert the distance score in a similarity score
                    # score = 1.0 - np.exp(score)

                elif metric == "overlap":

                    score = LanguageModelProfiler.lm_overlap(lm1, lm2)

                sentence_scores.append(score)

                # if debug:
                #     print("\t\t\t\t\t{} ({})".format(sentence2, score))

            # sort sentence scores in ascending order
            sorted_sentence_scores = sorted(sentence_scores)[:topk]
            # sorted_sentence_scores = sorted(sentence_scores, reverse=True)[:topk]
            score = sorted_sentence_scores[0]
            if topk > 1:
                for i in range(1, len(sorted_sentence_scores)):
                    score -= (1 / (i + 1)) * sorted_sentence_scores[i]

            if debug:
                top_score = np.min(sentence_scores)
                # top_score = np.max(sentence_scores)
                sentence_indexes_with_top_score = np.argwhere(sentence_scores == top_score).flatten().tolist()
                print("\t\tFound {} similar sentences in text2.".format(len(sentence_indexes_with_top_score)))

                if len(sentence_indexes_with_top_score) > 0:
                    for sentence_index_with_top_score in sentence_indexes_with_top_score:
                        sentence_with_top_score = text2[sentence_index_with_top_score]
                        print("\t\t\tSENTENCE {} (from text2): {} (KL={})".format(sentence_index_with_top_score,
                                                                                  sentence_with_top_score, score))
                else:
                    print("\t\t\tSCORES: {}".format(sentence_scores))

            scores.append(score)

        return scores

    @staticmethod
    def get_dataset_language_model_distance(data1, data2, topk=1, debug=False):
        """
        This method compares the similarity of two dataframes by measuring the KL divergence between the language models
        built over each pair of records from data1 and data2. This evaluation is performed into two directions: from
        data1 to data2 and from data2 to data1.

        :param data1: Pandas dataframe containing the first dataset
        :param data2: Pandas dataframe containing the second dataset
        :param topk: the optional number of top-k records to be considered to associate to each record a single
                     similarity score
        :param debug: flag for enabling the debug modality
        :return: two lists containing the similarity scores between each record of the first dataframe with respect the
                 second one (and viceversa)
        """

        if not isinstance(data1, pd.DataFrame):
            raise TypeError("Wrong data type for parameter data1. Only Pandas dataframe data type is allowed.")

        if not isinstance(data2, pd.DataFrame):
            raise TypeError("Wrong data type for parameter data2. Only Pandas dataframe data type is allowed.")

        if not isinstance(topk, int):
            raise TypeError("Wrong data type for parameter topk. Only integer data type is allowed.")

        max_data_len = np.max([len(data1), len(data2)])
        if topk <= 0 or topk > max_data_len:
            raise TypeError(
                "Wrong data value for parameter topk. Only values in the range [1, {}] are allowed.".format(
                    max_data_len))

        # convert input dataframes into clean text
        attrs1 = data1.columns.values
        clean_data1 = TextPreparation.convert_dataframe_to_text(data1, attrs1)
        attrs2 = data2.columns.values
        clean_data2 = TextPreparation.convert_dataframe_to_text(data2, attrs2)
        # metric = "kl"
        metric = "overlap"

        # direction data1 to data2
        print("\t\t\tDATA1 -> DATA2")
        data1_to_data2_lm_distance = LanguageModelProfiler.compare_text_language_models(clean_data1, clean_data2,
                                                                                        metric, topk=topk, debug=debug)

        # direction data2 to data1
        print("\t\t\tDATA2 -> DATA1")
        data2_to_data1_lm_distance = LanguageModelProfiler.compare_text_language_models(clean_data2, clean_data1,
                                                                                        metric, topk=topk, debug=debug)

        return data1_to_data2_lm_distance, data2_to_data1_lm_distance


class LanguageModel(object):
    def __init__(self, n, mtype='mle'):
        self.n = n
        self.model = self._get_language_model(mtype)

    def _get_language_model(self, mtype):
        if mtype == 'mle':
            return MLE(self.n)
        elif mtype == 'laplace':
            return Laplace(self.n)
        elif mtype == 'lidstone':
            return Lidstone(self.n)
        elif mtype == 'interpolated':
            return InterpolatedLanguageModel(self.n)
        elif mtype == 'written':
            return WittenBellInterpolated(self.n)
        else:
            raise ValueError('{} language model doesnt exists'.format(mtype))

    def fit(self, docs):
        # print('number of docs', len(docs))
        train, vocab = padded_everygram_pipeline(self.n, docs)
        # print('train, ', train)
        # print
        self.model.fit(train, vocab)

    def score_word(self, word):
        return self.model.score(word)

    def count_word(self, word):
        return self.model.counts[word]

    def get_vocabs(self):
        return list(self.model.vocab)

    def get_probs(self):
        vocabs = self.get_vocabs()
        scores = [self.score_word(word) for word in vocabs]
        return vocabs, scores

    def get_distribution(self):
        vocabs = self.get_vocabs()
        scores = [self.model.counts[word] for word in vocabs]
        return vocabs, scores

    def kl_smoothed(self, lm2, lm_collection=None, lambd=0.8):
        # get current vocab scores
        vocabs1, scores1 = self.get_probs()

        if lm_collection:
            # smoothed scores
            scores1 = [lambd * score + (1 - lambd) * lm_collection.score_word(vocab) for vocab, score in
                       zip(vocabs1, scores1)]

        # convert to numpy array
        a = np.asarray(scores1, dtype=np.float)

        # compute matched language model scores
        if lm_collection:
            b = [lambd * lm2.score_word(vocab) + (1 - lambd) * lm_collection.score_word(vocab) for vocab in vocabs1]
        else:
            b = [lm2.score_word(w) for w in vocabs1]

        # convert into numpy array
        b = np.asarray(b, dtype=np.float)

        return entropy(a, b)

    def histogram_intersection(self, lm2):
        vocabs, scores = self.get_distribution()
        s1 = pd.Series(scores, index=vocabs)

        vocabs, scores = lm2.get_distribution()
        s2 = pd.Series(scores, index=vocabs)

        ds = pd.concat([s1, s2], axis=1)
        ds.fillna(0, inplace=True)

        s1 = ds[0].values
        s2 = ds[1].values

        minima = np.minimum(s1, s2)
        intersection = np.true_divide(np.sum(minima), np.sum(s1))
        return intersection

    def scaled_histogram_intersection(self, lm2, alpha=1):
        vocabs, scores = self.get_distribution()
        s1 = pd.Series(scores, index=vocabs)

        vocabs, scores = lm2.get_distribution()
        s2 = pd.Series(scores, index=vocabs)

        ds = pd.concat([s1, s2], axis=1)
        ds.fillna(0, inplace=True)

        s1 = ds[0].values
        s2 = ds[1].values

        minima = np.minimum(s1, s2)
        intersection = np.true_divide(np.sum(minima), np.sum(s1))

        intersection = intersection * alpha * (len(self.get_vocabs()) / len(lm2.get_vocabs()))

        return intersection

    def histogram_overlap(self, lm2):
        vocabs, scores = self.get_distribution()
        s1 = pd.Series(scores, index=vocabs)

        vocabs, scores = lm2.get_distribution()
        s2 = pd.Series(scores, index=vocabs)

        ds = pd.concat([s1, s2], axis=1)
        ds.fillna(0, inplace=True)

        cond = (ds[0] != 0) & (ds[1] != 0)

        s1 = ds.loc[cond, 0].values
        s2 = ds.loc[cond, 1].values

        minima = np.minimum(s1, s2)
        intersection = np.true_divide(np.sum(minima), np.sum(s1))
        return intersection

    def histogram_log_difference(self, lmi):
        vocabs, scores = self.get_distribution()

        # convert scores1 to numpy array
        s1 = np.asarray(scores, dtype=np.float)

        # get score from integration language model
        s2 = [lmi.count_word(w) for w in vocabs]
        s2 = np.asarray(s2, dtype=np.float)

        # compute minimum intersection
        minima = np.minimum(s1, s2)

        # compute difference
        s_diff = s1 - minima
        i_diff = s2 - minima

        s_diff = s_diff + 1
        s_diff = np.log(s_diff)

        i_diff = i_diff + 1
        i_diff = np.log(i_diff)

        return np.mean(s_diff), np.mean(i_diff)

    def histogram_difference(self, lmi, num_sources):
        vocabs, scores = self.get_distribution()

        # convert scores1 to numpy array
        vocabs = np.asarray(vocabs)
        s1 = np.asarray(scores, dtype=np.float)

        # remove zero score
        vocabs = vocabs[s1 != 0]
        s1 = s1[s1 != 0]

        # get score from integration language model
        s2 = [lmi.count_word(w) for w in vocabs]
        s2 = np.asarray(s2, dtype=np.float)

        # compute minimum and maxima intersection
        minima = np.minimum(s1, s2)
        maxima = np.maximum(s1, s2)

        # compute difference
        s_diff = s1 - minima
        i_diff = s2 - minima

        # normalize
        s_diff = 1 - s_diff / maxima
        i_diff = 1 - i_diff / maxima

        x = np.mean(s_diff)
        y = np.mean(i_diff)
        # range_mapper = interp1d([1.0/num_sources, 1], [0, 1])
        # normalized_y = range_mapper(y)

        # return x, normalized_y
        return x, y

    def histogram_difference_sum(self, lmi):
        vocabs, scores = self.get_distribution()

        # convert scores1 to numpy array
        vocabs = np.asarray(vocabs)
        s1 = np.asarray(scores, dtype=np.float)

        # remove zero score
        vocabs = vocabs[s1 != 0]
        s1 = s1[s1 != 0]

        # get score from integration language model
        s2 = [lmi.count_word(w) for w in vocabs]
        s2 = np.asarray(s2, dtype=np.float)

        # compute minimum and maxima intersection
        minima = np.minimum(s1, s2)
        maxima = np.maximum(s1, s2)

        # compute difference
        s_diff = s1 - minima
        i_diff = s2 - minima

        # normalize
        s_diff = 1 - np.sum(s_diff) / np.sum(maxima)
        i_diff = 1 - np.sum(i_diff) / np.sum(maxima)

        return s_diff, i_diff

    def precision(self, lm2):
        # get distribution
        vocabs, scores = self.get_distribution()
        s1 = pd.Series(scores, index=vocabs)

        # get target distribution
        vocabs, scores = lm2.get_distribution()
        s2 = pd.Series(scores, index=vocabs)

        # overlap two language models
        ds = pd.concat([s1, s2], axis=1)
        ds.fillna(0, inplace=True)

        # keep only intersection
        cond = (ds[0] != 0) & (ds[1] != 0)

        s1 = ds.loc[cond, 0].values
        s2 = ds.loc[cond, 1].values

        # compute minimum and maxima intersection
        minima = np.minimum(s1, s2)
        maxima = np.maximum(s1, s2)

        # compute difference
        s_diff = s1 - minima
        i_diff = s2 - minima

        # normalize
        s_diff = 1 - s_diff / maxima
        i_diff = 1 - i_diff / maxima

        return np.mean(s_diff), np.mean(i_diff)

    def recall_vocab(self, lmi):
        # get distribution
        vocabs, scores = self.get_distribution()

        # convert scores1 to numpy array
        vocabs = np.asarray(vocabs)
        s1 = np.asarray(scores, dtype=np.float)

        # remove zero score
        vocabs = vocabs[s1 != 0]
        s1 = s1[s1 != 0]

        # get score from integration language model
        s2 = [lmi.count_word(w) for w in vocabs]
        s2 = np.asarray(s2, dtype=np.float)

        intersection = len(s1[s2 > 0])

        recall = intersection / len(s1)

        return recall

    def recall(self, lmi):
        # get distribution
        vocabs, scores = self.get_distribution()

        # convert scores1 to numpy array
        vocabs = np.asarray(vocabs)
        s1 = np.asarray(scores, dtype=np.float)

        # remove zero score
        vocabs = vocabs[s1 != 0]
        s1 = s1[s1 != 0]

        # get score from integration language model
        s2 = [lmi.count_word(w) for w in vocabs]
        s2 = np.asarray(s2, dtype=np.float)

        intersection = s1[s2 > 0].sum()

        recall = intersection / s1.sum()

        return recall


def get_distribution(records):
    # tokenize records
    # docs = [x.split() for x in records]
    docs = records
    # create language model
    lm = LanguageModel(n=1, model='mle')
    lm.fit(docs)

    vocabs, scores = lm.get_distribution()
    return pd.Series(scores, index=vocabs)
