import gensim.downloader as api
import spacy
import spacy.cli
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from uedi.utils.general_utilities import check_parameter_type
from uedi.data_integration.data_integration_utilities import check_document_format, check_tokenized_dataset_format


class GensimEmbeddingManager(object):
    """
    This class implements a Gensim word embedding manager.
    """

    def __init__(self, pretrained_model: str):
        """
        This method initializes the embedding manager state.

        :param pretrained_model: name of the pre-trained model to retrieve
        """

        check_parameter_type(pretrained_model, 'pretrained_model', str, 'string')

        available_models = api.info()["models"]
        if pretrained_model not in available_models:
            raise ValueError(
                "Wrong pretrained model name. Available models in Gensim are: {}".format(available_models))

        model = api.load(pretrained_model)
        self.model = model
        self.model_name = pretrained_model
        self.oov_map = {}

    def model_to_string(self):
        """
        This method creates a string version of the embedding manager object.
        """
        return "{}_{}".format(type(self).__name__, self.model_name.split('-')[0])

    def document_embedding(self, doc: list, dataset_id: str):
        """
        This method computes the embedding of a document by averaging the non-oov word embedding.

        :param doc: list of tokens
        :param dataset_id: the identifier of the dataset that contains the document
        :return: embedded representation of the input document
        """

        check_document_format(doc, 'doc')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        vector = None
        # remove out-of-vocabulary words (OOV)
        original_doc = doc[:]
        doc = [word for word in doc if word in self.model.vocab]

        removed_tokens = len(original_doc) - len(doc)
        if dataset_id not in self.oov_map:
            self.oov_map[dataset_id] = {'removed_tokens': removed_tokens}
        else:
            self.oov_map[dataset_id]['removed_tokens'] += removed_tokens

        if len(doc) == 0:
            return vector
        vector = np.mean(self.model[doc], axis=0)

        return vector

    def document_oov(self, doc: list):
        """
        This method finds the model out-of-vocabulary words included the provided document.

        :param doc: list of tokens
        :return: (number of out-of-vocabulary words, list of out-of-vocabulary words)
        """

        check_document_format(doc, 'doc')

        oov_words = [word for word in doc if word not in self.model.vocab]

        return len(oov_words), oov_words

    def dataset_oov(self, dataset: np.ndarray):
        """
        This method finds the model out-of-vocabulary words included the provided dataset.

        :param dataset: list of lists of tokens
        :return: (percentage of out-of-vocabulary words, list of out-of-vocabulary words)
        """

        check_tokenized_dataset_format(dataset, 'dataset')

        total_words = 0
        num_oov_words = 0
        oov_words = set([])

        for doc in dataset:
            num_doc_oov_words, doc_oov_words = self.document_oov(doc)

            total_words += len(doc)
            num_oov_words += num_doc_oov_words
            oov_words.update(doc_oov_words)

        return (num_oov_words / total_words) * 100, oov_words

    def dataset_embeddings(self, dataset: np.ndarray, dataset_id: str):
        """
        This method embeds a dataset based on the provided pre-trained word embedding model.

        :param dataset: list of lists of tokens
        :param dataset_id: the identifier of the dataset
        :return: embedded dataset
        """

        check_tokenized_dataset_format(dataset, 'dataset')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        embedding_matrix = []

        for doc in dataset:

            emb_doc = self.document_embedding(doc, dataset_id)
            if emb_doc is not None:
                embedding_matrix.append(emb_doc)

        removed_docs = len(dataset) - len(embedding_matrix)
        if dataset_id not in self.oov_map:
            self.oov_map[dataset_id] = {'removed_docs': removed_docs}
        else:
            if 'removed_docs' not in self.oov_map[dataset_id]:
                self.oov_map[dataset_id]['removed_docs'] = removed_docs
            else:
                self.oov_map[dataset_id]['removed_docs'] += removed_docs

        print("Removed {} docs.".format(removed_docs))

        return embedding_matrix

    def compare_embedded_datasets(self, dataset1: np.ndarray, dataset2: np.ndarray, embedded_dataset1=None,
                                  embedded_dataset2=None):
        """
        This method embeds a dataset based on the provided pre-trained word embedding model.

        :param dataset1: list of lists of tokens
        :param dataset2: list of lists of tokens
        :param embedded_dataset1: optional dataset embedded version
        :param embedded_dataset2: optional dataset embedded version
        :return: list of maximum cosine similarities between each document of dataset1 and each document of dataset2
        """

        check_tokenized_dataset_format(dataset1, 'dataset1')
        check_tokenized_dataset_format(dataset2, 'dataset2')
        check_parameter_type(embedded_dataset1, 'embedded_dataset1', np.ndarray, 'Numpy array', optional_param=True)
        check_parameter_type(embedded_dataset2, 'embedded_dataset2', np.ndarray, 'Numpy array', optional_param=True)
        # if embedded_dataset1 is None and embedded_dataset2 is None:
        #     if dataset1 is None or dataset2 is None:
        #         raise ValueError("Provide the original dataset contents or their embedded versions.")

        if embedded_dataset1 is None:
            print("Projecting dataset1 to the embedded space...")
            embedded_dataset1 = self.dataset_embeddings(dataset1, 'd1')
            print("Done.")

            print("Projecting dataset2 to the embedded space...")
            embedded_dataset2 = self.dataset_embeddings(dataset2, 'd2')
            print("Done.")

            embedded_dataset1 = np.array(embedded_dataset1)
            embedded_dataset2 = np.array(embedded_dataset2)
            embedded_dataset1[embedded_dataset1 < 0] = 0
            embedded_dataset2[embedded_dataset2 < 0] = 0

        sims = cosine_similarity(embedded_dataset1, embedded_dataset2)

        return np.mean(np.max(sims, axis=1))


class SpacyEmbeddingManager(object):
    """
    This class implements a Spacy word embedding manager.
    """

    def __init__(self, pretrained_model: str):
        """
        This method initializes the embedding manager state.

        :param pretrained_model: name of the pre-trained model to retrieve
        """

        check_parameter_type(pretrained_model, 'pretrained_model', str, 'string')

        available_models = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        if pretrained_model not in available_models:
            raise ValueError(
                "Wrong pretrained model name. Available models in Spacy are: {}".format(available_models))

        spacy.cli.download(pretrained_model)
        model = spacy.load(pretrained_model)
        self.model = model
        self.model_name = pretrained_model
        self.oov_map = {}

    def model_to_string(self):
        """
        This method creates a string version of the embedding manager object.
        """
        return "{}_{}".format(type(self).__name__, self.model_name.split('_')[-1])

    def document_embedding(self, doc: list, dataset_id: str):
        """
        This method computes the embedding of a document by averaging the non-oov word embedding.

        :param doc: list of tokens
        :param dataset_id: the identifier of the dataset that contains the document
        :return: embedded representation of the input document
        """

        check_document_format(doc, 'doc')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        vector = None
        original_doc = doc[:]
        # remove out-of-vocabulary words (OOV)
        doc = [str(word) for word in self.model(' '.join(doc)) if not word.is_oov]

        removed_tokens = len(original_doc) - len(doc)
        if dataset_id not in self.oov_map:
            self.oov_map[dataset_id] = {'removed_tokens': removed_tokens}
        else:
            self.oov_map[dataset_id]['removed_tokens'] += removed_tokens

        if len(doc) == 0:
            return vector
        vector = self.model(' '.join(doc)).vector

        return vector

    def document_oov(self, doc: list):
        """
        This method finds the model out-of-vocabulary words included the provided document.

        :param doc: list of tokens
        :return: (number of out-of-vocabulary words, list of out-of-vocabulary words)
        """

        check_document_format(doc, 'doc')

        oov_words = [str(word) for word in self.model(' '.join(doc)) if word.is_oov]

        return len(oov_words), oov_words

    def dataset_oov(self, dataset: np.ndarray):
        """
        This method finds the model out-of-vocabulary words included the provided dataset.

        :param dataset: list of lists of tokens
        :return: (percentage of out-of-vocabulary words, list of out-of-vocabulary words)
        """

        check_tokenized_dataset_format(dataset, 'dataset')

        total_words = 0
        num_oov_words = 0
        oov_words = set([])

        for doc in dataset:
            num_doc_oov_words, doc_oov_words = self.document_oov(doc)

            total_words += len(doc)
            num_oov_words += num_doc_oov_words
            oov_words.update(doc_oov_words)

        return (num_oov_words / total_words) * 100, oov_words

    def dataset_embeddings(self, dataset: np.ndarray, dataset_id: str):
        """
        This method embeds a dataset based on the provided pre-trained word embedding model.

        :param dataset: list of lists of tokens
        :param dataset_id: the identifier of the dataset
        :return: embedded dataset
        """

        check_tokenized_dataset_format(dataset, 'dataset')
        check_parameter_type(dataset_id, 'dataset_id', str, 'string')

        embedding_matrix = []

        for doc in dataset:

            emb_doc = self.document_embedding(doc, dataset_id)
            if emb_doc is not None:
                embedding_matrix.append(emb_doc)

        removed_docs = len(dataset) - len(embedding_matrix)
        if dataset_id not in self.oov_map:
            self.oov_map[dataset_id] = {'removed_docs': removed_docs}
        else:
            if 'removed_docs' not in self.oov_map[dataset_id]:
                self.oov_map[dataset_id]['removed_docs'] = removed_docs
            else:
                self.oov_map[dataset_id]['removed_docs'] += removed_docs

        print("Removed {} docs.".format(removed_docs))

        return embedding_matrix

    def compare_embedded_datasets(self, dataset1: np.ndarray, dataset2: np.ndarray, embedded_dataset1=None,
                                  embedded_dataset2=None):
        """
        This method embeds a dataset based on the pre-trained word embedding model.

        :param dataset1: list of lists of tokens
        :param dataset2: list of lists of tokens
        :param embedded_dataset1: optional dataset embedded version
        :param embedded_dataset2: optional dataset embedded version
        :return: list of maximum cosine similarities between each document of dataset1 and each document of dataset2
        """

        check_tokenized_dataset_format(dataset1, 'dataset1')
        check_tokenized_dataset_format(dataset2, 'dataset2')
        check_parameter_type(embedded_dataset1, 'embedded_dataset1', np.ndarray, 'Numpy array', optional_param=True)
        check_parameter_type(embedded_dataset2, 'embedded_dataset2', np.ndarray, 'Numpy array', optional_param=True)
        # if embedded_dataset1 is None and embedded_dataset2 is None:
        #     if dataset1 is None or dataset2 is None:
        #         raise ValueError("Provide the original dataset contents or their embedded versions.")

        if embedded_dataset1 is None:
            print("Projecting dataset1 to the embedded space...")
            embedded_dataset1 = self.dataset_embeddings(dataset1, 'd1')
            print("Done.")

            print("Projecting dataset2 to the embedded space...")
            embedded_dataset2 = self.dataset_embeddings(dataset2, 'd2')
            print("Done.")

            embedded_dataset1 = np.array(embedded_dataset1)
            embedded_dataset2 = np.array(embedded_dataset2)
            embedded_dataset1[embedded_dataset1 < 0] = 0
            embedded_dataset2[embedded_dataset2 < 0] = 0

        sims = cosine_similarity(embedded_dataset1, embedded_dataset2)

        return np.mean(np.max(sims, axis=1))
