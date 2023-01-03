import pandas as pd
from typing import List
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def split_row(df_row: (list, pd.Series), sep: str = ' ') -> List[str]:
    """This method split a Pandas DataFrame row.

    Args:
        df_row: Iterable object (i.e., list, pd.Series) containing the row of a Pandas DataFrame
        sep: The string separator to be used to split values inside the same attribute

    Returns:
        list of strings corresponding to the Pandas DataFrame separated row
    """
    tokenized_row = []
    # loop over row attributes
    for attr_val in df_row:
        attr_val = str(attr_val)
        # split attribute value
        tokenized_row += attr_val.strip().split(sep)
    return tokenized_row


def lowercase_tokens(word_tokens: List[str]) -> List[str]:
    """This method lowercase word tokens.

    Args: word_tokens: list of word tokens

    Returns:
        List of strings containing lowercase word tokens
    """

    # lowercase all words
    return [word.lower() for word in word_tokens]


def remove_punctuation(word_tokens: List[str]) -> List[str]:
    """This method removes punctuation from word tokens.

    Args:
        word_tokens: list of word tokens

    Returns:
        List of strings containing non-punctuation tokens
    """

    # filter out punctuation
    filtered_words = [w for w in word_tokens if w not in punctuation]

    return filtered_words


def remove_stop_words(word_tokens: List[str]) -> List[str]:
    """This method removes stop words from word tokens.

    Args:
        word_tokens: list of word tokens

    Returns:
        List of strings containing non-stop-word tokens
    """

    # set of stop words
    stop_words = set(stopwords.words('english'))

    # filter out stop words
    filtered_words = [w for w in word_tokens if w not in stop_words]

    return filtered_words


def lemmatization(word_tokens: List[str], ) -> List[str]:
    """This method applies lemmatization to word tokens.
    Args:
        word_tokens: list of word tokens

    Returns:
        List of strings containing word tokens subjected to lemmatization
    """
    lemma_words = []

    lemmatizer = WordNetLemmatizer()

    for w in word_tokens:
        word1 = lemmatizer.lemmatize(w, pos="n")
        word2 = lemmatizer.lemmatize(word1, pos="v")
        word3 = lemmatizer.lemmatize(word2, pos=("a"))
        lemma_words.append(word3)

    return lemma_words


def stemming(word_tokens: List[str], ) -> List[str]:
    """This method applies stemming to word tokens.
    Args:
        word_tokens: list of word tokens

    Returns:
        List of strings containing word tokens subjected to stemming
    """
    stem_words = []
    ps = PorterStemmer()
    for w in word_tokens:
        root_word = ps.stem(w)
        stem_words.append(root_word)

    return stem_words


def tokenizer(df_row: pd.Series, strategy: str = None) -> List[str]:
    """This method tokenizes a Pandas DataFrame row.

    Args:
        df_row:
            Iterable object (i.e., list, pd.Series) containing the row of a Pandas DataFrame
        strategy:
            String preprocessing strategy to be applied at each token.
            It could be stemming or lemmatization or None

    Returns:
        list of strings corresponding to the Pandas DataFrame tokenized row
    """
    tokens = split_row(df_row)
    tokens = lowercase_tokens(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_stop_words(tokens)
    if strategy == 'stemming':
        tokens = stemming(tokens)
    elif strategy == 'lemmatization':
        tokens = lemmatization(tokens)
    return tokens
