import spacy

from .base import Transformer


class Lemmatizer(Transformer):

    def __init__(self, model: str):
        self.model = model
        self.model_ = spacy.load(self.model)

    def transform(self, X, y=None):
        return [
            self._lemma(x) for x in X
        ]

    def _lemma(self, doc: str) -> str:
        return " "\
            .join([
                token.lemma_
                for
                    token in self.model_(doc)
            ])\
            .replace(" \n ", "\n")