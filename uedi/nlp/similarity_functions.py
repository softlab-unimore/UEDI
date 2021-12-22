import pandas as pd
import numpy as np
from uedi.nlp.pre_processing import TextPreparation
import collections


def jaccard_similarity(set1, set2):
    """
    Computes Jaccard measure.
    For two sets X and Y, the Jaccard similarity score is:
        :math:`jaccard(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|}`

    Note:
        In the case where both X and Y are empty sets, we define their Jaccard score to be 1.

    Args:
        set1,set2 (set, list or np.ndarray): Input sets (or lists/ndarrays). Input lists are converted to sets.
    Returns:
        Jaccard similarity score (float).
    Raises:
        TypeError : If the inputs are not sets (or lists/ndarrays) or if one of the inputs is None.
    """

    # input validations
    if set1 is None:
        raise TypeError("First argument cannot be None")

    if set2 is None:
        raise TypeError("Second argument cannot be None")

    if not isinstance(set1, (list, set, np.ndarray)):
        raise TypeError('First argument is expected to be a python list, set or numpy array')

    if not isinstance(set2, (list, set, np.ndarray)):
        raise TypeError('Second argument is expected to be a python list, set or numpy array')

    # if exact match return 1.0
    if set1 == set2:
        return 1.0

    # if one of the strings is empty return 0
    if len(set1) == 0 or len(set2) == 0:
        return 0

    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)

    return float(len(set1 & set2)) / float(len(set1 | set2))


class JaccardCalculator(object):
    """
    This class implements a series of dataset-level metrics based on the Jaccard similarity function.
    """

    @staticmethod
    def text_jaccard_similarity(text1, text2, topk=1, debug=False):
        """
        This method computes the Jaccard scores between two texts. The sentences of the first text are compared with
        the sentences of the second text and the Jaccard similarities are computed in order to discover the most
        similar sentences. A similarity score is assigned to each sentence of the first text with respect the second
        text. The final score is the maximum Jaccard similarity obtained or a sort of weighted mean calculated over
        the top-k most similar sentences.

        :param text1: list of sentences
        :param text2: list of sentences
        :param topk: the optional number of top-k most similar record to be considered for computing the Jaccard
                     weighted mean
        :param debug: flag for enabling the debug modality
        :return: a list of Jaccard similarity for each record of the first dataset
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

        if not isinstance(topk, int):
            raise TypeError("Wrong data type for parameter topk. Only integer data type is allowed.")

        if topk <= 0 or topk > len(text2):
            raise TypeError(
                "Wrong data value for parameter topk. Only values in the range [1, {}] are allowed.".format(
                    len(text2)))

        scores = []

        # loop over the sentences of the first text
        for sentence1_id, sentence1 in enumerate(text1):

            if debug:
                print("\tSENTENCE {} (from text1): {}".format(sentence1_id, sentence1))

            sentence_scores = []
            # loop over the sentences of the second text
            for sentence2_id, sentence2 in enumerate(text2):

                # compute jaccard similarity
                score = jaccard_similarity(sentence1, sentence2)

                sentence_scores.append(score)

            # sort sentence scores in ascending order
            sorted_sentence_scores = sorted(sentence_scores, reverse=True)[:topk]
            score = sorted_sentence_scores[0]
            if topk > 1:
                for i in range(1, len(sorted_sentence_scores)):
                    score -= (1 / (i + 1)) * sorted_sentence_scores[i]

            if debug:
                top_score = np.max(sentence_scores)
                sentence_indexes_with_top_score = np.argwhere(sentence_scores == top_score).flatten().tolist()
                print("\t\tFound {} similar sentences in text2.".format(len(sentence_indexes_with_top_score)))

                if len(sentence_indexes_with_top_score) > 0:
                    for sentence_index_with_top_score in sentence_indexes_with_top_score:
                        sentence_with_top_score = text2[sentence_index_with_top_score]
                        print("\t\t\tSENTENCE {} (from text2): {} (JACCARD={})".format(sentence_index_with_top_score,
                                                                                  sentence_with_top_score, score))
                else:
                    print("\t\t\tSCORES: {}".format(sentence_scores))

            scores.append(score)

        return scores

    @staticmethod
    def get_datasets_Jaccard(data1, data2, topk=1, debug=False):
        """
        This method computes the Jaccard similarities between two datasets. The records of the two datasets are compared
        in order to discover their reciprocal similarity. First the records of the first dataset are compared with the
        second one, and then the roles are reversed.

        :param data1: Pandas dataframe containing the first dataset
        :param data2: Pandas dataframe containing the second dataset
        :param topk: the optional number of top-k most similar record to be considered for computing the Jaccard
                     weighted mean
        :param debug: flag for enabling the debug modality
        :return: two lists containing the Jaccard similarities between each records of the two datasets
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

        # direction data1 to data2
        data1_to_data2_jaccard = JaccardCalculator.text_jaccard_similarity(clean_data1, clean_data2, topk=topk, debug=debug)

        # direction data2 to data1
        data2_to_data1_jaccard = JaccardCalculator.text_jaccard_similarity(clean_data2, clean_data1, topk=topk, debug=debug)

        return data1_to_data2_jaccard, data2_to_data1_jaccard
