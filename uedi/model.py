from typing import Literal, List
import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import MLE, Laplace, Lidstone, WittenBellInterpolated


class LanguageModel(object):
    def __init__(self, n: int, model: Literal['mle', 'laplace', 'lidstone', 'written'] = 'mle'):
        self.n = n
        self.model = self._get_language_model(model)

    def _get_language_model(self, mtype):
        if mtype == 'mle':
            return MLE(self.n)
        elif mtype == 'laplace':
            return Laplace(self.n)
        elif mtype == 'lidstone':
            return Lidstone(self.n)
        elif mtype == 'written':
            return WittenBellInterpolated(self.n)
        else:
            raise ValueError('{} language model doesnt exists'.format(mtype))

    def fit(self, docs: List[List[str]]):
        train, vocab = padded_everygram_pipeline(self.n, docs)
        self.model.fit(train, vocab)

    def score_word(self, word: str) -> float:
        return self.model.score(word)

    def count_word(self, word: str) -> int:
        return self.model.counts[word]

    def get_vocabs(self) -> List[str]:
        return list(self.model.vocab)

    def get_probs(self) -> [List[str], List[float]]:
        vocabs = self.get_vocabs()
        scores = [self.score_word(word) for word in vocabs]
        return vocabs, scores

    def get_distribution(self) -> [List[str], List[int]]:
        vocabs = self.get_vocabs()
        scores = [self.model.counts[word] for word in vocabs]
        return vocabs, scores


def get_distribution(records: List[List[str]]) -> pd.Series:
    # tokenize records
    # docs = [x.split() for x in records]
    # docs = records
    # create language model
    lm = LanguageModel(n=1, model='mle')
    lm.fit(records)

    vocabs, scores = lm.get_distribution()
    return pd.Series(scores, index=vocabs)
