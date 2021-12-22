import numpy as np
import pandas as pd
from gensim import corpora
import math
from uedi.nlp.pre_processing import TextPreparation
import collections


class BM25:
    """
    This class implements the BM25 scoring function.
    """

    def __init__(self, fn_docs, delimiter='|'):
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []

        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self):
        raw_data = []
        for line in self.fn_docs:
            # raw_data.append(line.strip().split(self.delimiter))
            raw_data.append(line)
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e):
        docTotalLen = 0
        for line in self.fn_docs:
            # doc = line.strip().split(self.delimiter)
            doc = line
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            # print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq * 1.0 / len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items():
                if term not in self.DF:
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)

            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] + 0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75):
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF):
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms:
                upper = (doc[term] * (k1 + 1))
                below = ((doc[term]) + k1 * (1 - b + b * doc_terms_len / self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self):
        tfidf = []
        for doc in self.DocTF:
            doc_tfidf = [(term, tf * self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self):
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items = list(items)
        items.sort()
        return items


class BM25Calculator(object):
    """
    This class implements a series of dataset-level metrics based on the BM25 scoring function.
    """

    @staticmethod
    def text_bm25(text1, text2, topk=1, debug=False):
        """
        This method computes the BM25 scores between two texts. The sentences of the first text are considered
        as queries, while the sentences of the second text are considered as documents (i.e., possible relevant
        results for a query).
        Each query is compared with all the documents in the collection and the BM25 scores are computed in order to
        discover the most relevant documents. The final score associated to the considered query is the maximum BM25
        score obtained or a sort of weighted mean calculated over the top-k most similar documents.

        :param text1: Pandas dataframe which contains the queries
        :param text2: Pandas dataframe which contains the collection of documents
        :param topk: the optional number of top-k most similar documents to be considered for computing the BM25
                     weighted mean
        :param debug: flag for enabling the debug modality
        :return: a list of BM25-based scores for each input query
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

        # create document collection
        bm25 = BM25(text2, delimiter=' ')

        if debug:
            sample_documents = text2[:5]
            print("\nCORPUS")
            for doc in sample_documents:
                print(doc)
            sample_bm25 = BM25(sample_documents, delimiter=' ')
            print("\nVOCABULARY")
            print(sample_bm25.Items())
            print("\nTFIDF")
            sample_tfidf = sample_bm25.TFIDF()
            for i, tfidfscore in enumerate(sample_tfidf):
                print(i, tfidfscore)

        # submit queries and compute the BM25-based measures
        bm25_scores = []
        for sentence1_id, sentence1 in enumerate(text1):

            if debug:
                print("\tSENTENCE {} (from text1): {}".format(sentence1_id, sentence1))

            scores = bm25.BM25Score(sentence1)

            sorted_scores = sorted(scores, reverse=True)[:topk]
            score = sorted_scores[0]
            if topk > 1:
                for i in range(1, len(sorted_scores)):
                    score -= (1 / (i + 1)) * sorted_scores[i]

            if debug:
                top_score = np.max(scores)
                sentence_indexes_with_top_score = np.argwhere(scores == top_score).flatten().tolist()
                print("\t\tFound {} similar sentences in text2.".format(len(sentence_indexes_with_top_score)))

                if len(sentence_indexes_with_top_score) > 0:
                    for sentence_index_with_top_score in sentence_indexes_with_top_score:
                        sentence_with_top_score = text2[sentence_index_with_top_score]
                        print("\t\t\tSENTENCE {} (from text2): {} (BM25={})".format(sentence_index_with_top_score,
                                                                                  sentence_with_top_score, score))
                else:
                    print("\t\t\tSCORES: {}".format(scores))

            bm25_scores.append(score)

        return bm25_scores

    @staticmethod
    def get_datasets_bm25(data1, data2, topk=1, debug=False):
        """
        This method computes the BM25 scores between two datasets. The records of the two datasets are considered
        interchangeably as queries and documents in a virtual information retrieval system. Initially, the records of
        the first dataset are considered as queries and the records of the second dataset are considered as documents
        (i.e., possible relevant results for a query). Then the roles are reversed.

        :param data1: Pandas dataframe which contains the queries
        :param data2: Pandas dataframe which contains the collection of documents
        :param topk: the optional number of top-k most similar documents to be considered for computing the BM25
                     weighted mean
        :param debug: flag for enabling the debug modality
        :return: a list of BM25-based scores for each input query
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
        data1_to_data2_bm25_scores = BM25Calculator.text_bm25(clean_data1, clean_data2, topk=topk, debug=debug)

        # direction data2 to data1
        data2_to_data1_bm25_scores = BM25Calculator.text_bm25(clean_data2, clean_data1, topk=topk, debug=debug)

        return data1_to_data2_bm25_scores, data2_to_data1_bm25_scores
