import json
import logging as log
from os.path import abspath, dirname, realpath

from langdetect import detect as lang_detect
from langdetect.detector import LangDetectException
from nltk.stem.snowball import SnowballStemmer

from .base import Transformer

with open(abspath(dirname(realpath(__file__))+"/iso639-1.json"), "r") as j:
    ISO639 = json.load(j)


class Stemmer(Transformer):

    def __init__(self,
        lang: str = None,
        ignore_startswith: str = "",
        ignore_stopwords: bool = True,
    ):
        self.lang = lang
        self.ignore_startswith = ignore_startswith
        self.ignore_stopwords = ignore_stopwords

    def transform(self, X, y=None):
        return [
            self._stem(
                sent,
                lang=lang,
                ignore_startswith=self.ignore_startswith,
                ignore_stopwords=self.ignore_stopwords,
            )
            for sent, lang in zip(
                X, (y if y is not None else [self.lang] * len(X))
            )
        ]

    def _stem(
        self,
        sentence: str,
        lang: str = None,
        ignore_startswith: str = "",
        ignore_stopwords: bool = True,
    ) -> str:
        lang = lang or self._lang(sentence)[1]

        if lang in SnowballStemmer.languages:
            stemmer = SnowballStemmer(
                language=lang,
                ignore_stopwords=ignore_stopwords
            )
            return "\n".join([
                " ".join([
                    stemmer.stem(w)
                    for
                        w in sent.split()
                    if
                        w[:1] not in ignore_startswith
                ])
                for sent in sentence.split("\n")
            ])

        log.debug(f"SnowballStemmer: '{lang}' not found. Skipping...")
        return sentence

    @staticmethod
    def _lang(sentence: str) -> list:
        lang = None
        try:
            lang = lang_detect(sentence)
        except LangDetectException as e:  # No detected language
            log.debug(f"LangDetectException: {e}.")
        return [lang, ISO639.get(lang)]