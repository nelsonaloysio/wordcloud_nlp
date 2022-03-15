import nltk

from .base import Transformer

class NGrams(Transformer):

    def __init__(self, n_grams: int):
        self.n_grams = n_grams

    def transform(self, X):
        return [
            self._ngrams(
                x.split() if type(x) == str else x,
                self.n_grams,
            )
            for x in X
        ]

    @staticmethod
    def _ngrams(tokens: list, n: int, func = lambda x: " ".join(list(x))):
        return [
            func(g)
            if
                func is not None
            else
                g
            for g in
                list(nltk.ngrams(tokens, n))
            if
                len(set(g)) == n
        ]