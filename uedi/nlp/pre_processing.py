from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import collections


class TextPreparation(object):
    """
    This class manages the preparation of textual data coming from different data sources (i.e., strings, Pandas
    DataFrame, etc).
    """

    @staticmethod
    def tokenize_text(text):
        """
        This method tokenizes text (i.e., string organized in multiple lines).

        :param text: the text to be tokenized
        :return: the tokenized text
        """

        if not isinstance(text, str):
            raise TypeError("Wrong data type for parameter text. Only string data type is allowed.")

        tokenized_text = [list(map(str.lower, word_tokenize(sent)))
                          for sent in sent_tokenize(text)]

        return tokenized_text

    @staticmethod
    def dataframe_tokenizer(df, columns, sep=' '):
        """
        This method tokenizes a Pandas DataFrame.

        :param df: Pandas DataFrame object
        :param columns: Pandas DataFrame target columns
        :param sep: the string separator to be used to split values inside Pandas DataFrame attributes
        :return: list of lists of strings corresponding to the tokenized Pandas DataFrame
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Wrong data type for parameter df. Only pd.DataFrame data type is allowed.")

        if not isinstance(columns, collections.Iterable):
            raise TypeError("Wrong data type for parameter columns. Only iterable data type is allowed.")

        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Wrong data type for columns elements. Only string data type is allowed.")

        if not isinstance(sep, str):
            raise TypeError("Wrong data type for parameter sep. Only string data type is allowed.")

        df_columns = df.columns.values
        for col in columns:
            if col not in df_columns:
                raise ValueError("No column {} found in the DataFrame.".format(col))

        selected_df = df[columns]

        tokenized_df = []
        for row_id, row in selected_df.iterrows():
            tokenized_row = TextPreparation.dataframe_row_tokenizer(row, sep=sep)
            tokenized_df.append(tokenized_row)

        return tokenized_df

    @staticmethod
    def dataframe_row_tokenizer(df_row, sep=' '):
        """
        This method tokenizes a Pandas DataFrame row.

        :param df_row: iterable object (i.e., list, pd.Series) containing the row of a Pandas DataFrame
        :param sep: the string separator to be used to split values inside the same attribute
        :return: list of strings corresponding to the Pandas DataFrame tokenized row
        """

        if not isinstance(df_row, collections.Iterable):
            raise TypeError("Wrong data type for parameter df_row. Only iterable data type is allowed.")

        if not isinstance(sep, str):
            raise TypeError("Wrong data type for parameter sep. Only string data type is allowed.")

        tokenized_row = []
        # loop over row attributes
        for attr_val in df_row:
            attr_val = str(attr_val)
            # split attribute value
            tokenized_row += attr_val.strip().split(sep)

        return tokenized_row

    @staticmethod
    def remove_single_character_tokens(word_tokens):
        """
        This method removes single-character tokens (mostly punctuation) from word tokens.

        :param word_tokens: word tokens
        :return: list of strings containing word tokens except single-character tokens
        """

        if not isinstance(word_tokens, collections.Iterable):
            raise TypeError("Wrong data type for parameter word_tokens. Only iterable data type is allowed.")

        for token in word_tokens:
            if not isinstance(token, str):
                raise TypeError("Wrong data type for word_tokens elements. Only string data type is allowed.")

        return [word for word in word_tokens if len(word) > 1]

    @staticmethod
    def remove_single_character_tokens_from_text(tokenized_text):
        """
        This method removes single-character tokens from tokenized text.

        :param tokenized_text: tokenized text
        :return: list of lists of strings containing word tokens except single-character tokens
        """

        if not isinstance(tokenized_text, collections.Iterable):
            raise TypeError("Wrong data type for parameter tokenized_text. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            if not isinstance(word_tokens, collections.Iterable):
                raise TypeError("Wrong data type for word tokens. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            for token in word_tokens:
                if not isinstance(token, str):
                    raise TypeError("Wrong data type for tokens. Only string data type is allowed.")

        # filter out single-character tokens
        filtered_text = []
        for word_tokens in tokenized_text:
            filtered_words = TextPreparation.remove_single_character_tokens(word_tokens)

            filtered_text.append(filtered_words)

        return filtered_text

    @staticmethod
    def lowercase_tokens(word_tokens):
        """
        This method lowercase word tokens.

        :param word_tokens: word tokens
        :return: list of strings containing lowercase word tokens
        """

        if not isinstance(word_tokens, collections.Iterable):
            raise TypeError("Wrong data type for parameter word_tokens. Only iterable data type is allowed.")

        for token in word_tokens:
            if not isinstance(token, str):
                raise TypeError("Wrong data type for word_tokens elements. Only string data type is allowed.")

        # lowercase all words
        return [word.lower() for word in word_tokens]

    @staticmethod
    def lowercase_text(tokenized_text):
        """
        This method lowercase tokens from tokenized text.

        :param tokenized_text: tokenized text
        :return: list of lists of strings containing lowercase word tokens
        """

        if not isinstance(tokenized_text, collections.Iterable):
            raise TypeError("Wrong data type for parameter tokenized_text. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            if not isinstance(word_tokens, collections.Iterable):
                raise TypeError("Wrong data type for word tokens. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            for token in word_tokens:
                if not isinstance(token, str):
                    raise TypeError("Wrong data type for tokens. Only string data type is allowed.")

        # lowercase tokens
        lowercase_text = []
        for word_tokens in tokenized_text:
            lowercase_words = TextPreparation.lowercase_tokens(word_tokens)

            lowercase_text.append(lowercase_words)

        return lowercase_text

    @staticmethod
    def remove_stop_words(word_tokens):
        """
        This method removes stop words from word tokens.

        :param word_tokens: word tokens
        :return: list of strings containing non-stop-word tokens
        """

        if not isinstance(word_tokens, collections.Iterable):
            raise TypeError("Wrong data type for parameter word_tokens. Only iterable data type is allowed.")

        for token in word_tokens:
            if not isinstance(token, str):
                raise TypeError("Wrong data type for word_tokens elements. Only string data type is allowed.")

        # set of stop words
        stop_words = set(stopwords.words('english'))

        # filter out stop words
        filtered_words = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_words.append(w)

        return filtered_words

    @staticmethod
    def remove_stop_words_from_text(tokenized_text):
        """
        This method removes stop words from tokenized text.

        :param tokenized_text: tokenized text
        :return: list of lists of strings containing non-stop-word tokens
        """

        if not isinstance(tokenized_text, collections.Iterable):
            raise TypeError("Wrong data type for parameter tokenized_text. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            if not isinstance(word_tokens, collections.Iterable):
                raise TypeError("Wrong data type for word tokens. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            for token in word_tokens:
                if not isinstance(token, str):
                    raise TypeError("Wrong data type for tokens. Only string data type is allowed.")

        # filter out stop words
        filtered_text = []
        for word_tokens in tokenized_text:
            filtered_words = TextPreparation.remove_stop_words(word_tokens)

            filtered_text.append(filtered_words)

        return filtered_text

    @staticmethod
    def word_stemming(word_tokens):
        """
        This method stems word tokens.

        :param word_tokens: word tokens to be stemmed
        :return: list of string containing stemmed word tokens
        """

        if not isinstance(word_tokens, collections.Iterable):
            raise TypeError("Wrong data type for parameter word_tokens. Only iterable data type is allowed.")

        for token in word_tokens:
            if not isinstance(token, str):
                raise TypeError("Wrong data type for word_tokens elements. Only string data type is allowed.")

        stem_words = []
        ps = PorterStemmer()
        for w in word_tokens:
            root_word = ps.stem(w)
            stem_words.append(root_word)

        return stem_words

    @staticmethod
    def text_stemming(tokenized_text):
        """
        This method stems tokenized text.

        :param tokenized_text: tokenized text to be stemmed
        :return: list of lists of strings containing stemmed word tokens
        """

        if not isinstance(tokenized_text, collections.Iterable):
            raise TypeError("Wrong data type for parameter tokenized_text. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            if not isinstance(word_tokens, collections.Iterable):
                raise TypeError("Wrong data type for word tokens. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            for token in word_tokens:
                if not isinstance(token, str):
                    raise TypeError("Wrong data type for tokens. Only string data type is allowed.")

        stemmed_text = []
        for word_tokens in tokenized_text:
            stemmed_words = TextPreparation.word_stemming(word_tokens)

            stemmed_text.append(stemmed_words)

        return stemmed_text

    @staticmethod
    def word_lemmatization(word_tokens):
        """
        This method applies lemmatization to word tokens.

        :param word_tokens: word tokens where to apply lemmatization
        :return: list of strings containing word tokens subjected to lemmatization
        """

        if not isinstance(word_tokens, collections.Iterable):
            raise TypeError("Wrong data type for parameter word_tokens. Only iterable data type is allowed.")

        for token in word_tokens:
            if not isinstance(token, str):
                raise TypeError("Wrong data type for word_tokens elements. Only string data type is allowed.")

            lemma_words = []

            wordnet_lemmatizer = WordNetLemmatizer()

            for w in word_tokens:
                word1 = wordnet_lemmatizer.lemmatize(w, pos="n")
                word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
                word3 = wordnet_lemmatizer.lemmatize(word2, pos=("a"))
                lemma_words.append(word3)

            return lemma_words

    @staticmethod
    def text_lemmatization(tokenized_text):
        """
        This method applies lemmatization to tokenized text.

        :param tokenized_text: tokenized text where to apply lemmatization
        :return: list of lists of strings containing word tokens subjected to lemmatization
        """

        if not isinstance(tokenized_text, collections.Iterable):
            raise TypeError("Wrong data type for parameter tokenized_text. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            if not isinstance(word_tokens, collections.Iterable):
                raise TypeError("Wrong data type for word tokens. Only iterable data type is allowed.")

        for word_tokens in tokenized_text:
            for token in word_tokens:
                if not isinstance(token, str):
                    raise TypeError("Wrong data type for tokens. Only string data type is allowed.")

        lamma_text = []
        for word_tokens in tokenized_text:
            lemma_words = TextPreparation.word_lemmatization(word_tokens)

            lamma_text.append(lemma_words)

        return lamma_text

    @staticmethod
    def convert_dataframe_to_text(df, columns, sep=' ', stemming_type=None):
        """
        This method convert a Pandas DataFrame object to tokenized text.

        :param df: Pandas DataFrame object
        :param columns: Pandas DataFrame target columns
        :param sep: the string separator to be used to split values inside Pandas DataFrame attributes
        :param stemming_type: type of stemming to apply
        :return: list of lists of strings corresponding to the tokenized Pandas DataFrame
        """

        # check data types
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Wrong data type for parameter df. Only pd.DataFrame data type is allowed.")

        if not isinstance(columns, collections.Iterable):
            raise TypeError("Wrong data type for parameter columns. Only iterable data type is allowed.")

        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Wrong data type for columns elements. Only string data type is allowed.")

        if not isinstance(sep, str):
            raise TypeError("Wrong data type for parameter sep. Only string data type is allowed.")

        if stemming_type is not None:
            if not isinstance(stemming_type, str):
                raise TypeError("Wrong data type for parameter stemming_type. Only string data type is allowed.")

        # check data values
        df_columns = df.columns.values
        for col in columns:
            if col not in df_columns:
                raise ValueError("No column {} found in the DataFrame.".format(col))

        if stemming_type:
            stemming_types = ["stemming", "lemmatization"]
            if stemming_type not in stemming_types:
                raise ValueError(
                    "Wrong data value for parameter stemming type. Only values {} are allowed.".format(stemming_types))

        tokenized_df = TextPreparation.dataframe_tokenizer(df, columns, sep)

        # filtered_tokenized_df = TextPreparation.remove_single_character_tokens_from_text(tokenized_df)

        lowercase_tokenized_df = TextPreparation.lowercase_text(tokenized_df)

        tokenized_df_no_stop = TextPreparation.remove_stop_words_from_text(lowercase_tokenized_df)

        if stemming_type:
            if stemming_type == "stemming":

                stemmed_df = TextPreparation.text_stemming(tokenized_df_no_stop)

            else:  # lemmatization

                stemmed_df = TextPreparation.text_lemmatization(tokenized_df_no_stop)

        else:

            stemmed_df = tokenized_df_no_stop

        return stemmed_df