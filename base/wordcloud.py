import json
from os.path import abspath, dirname, isfile, realpath
from urllib.request import urlopen

import pandas as pd

from .base import Transformer

D3JS = abspath(dirname(realpath(__file__))+'/d3.layout.cloud.js')
D3HTML = abspath(dirname(realpath(__file__))+'/d3.layout.cloud.html')

URL = 'https://raw.githubusercontent.com/jasondavies/d3-cloud/master/build/d3.layout.cloud.js'

class Wordcloud(Transformer):

    def __init__(
        self,
        exclude_words: list = [],
        max_words: int = None,
    ):
        self.exclude_words = exclude_words
        self.max_words = max_words

    def transform(self, X):
        return self._wordcloud(
            self._wordcount(
                X,
                max_words=self.max_words,
                exclude_words=self.exclude_words,
            ),
        )

    def _wordcloud(self, dct: dict, render: bool = True):
        return self._render(self.__normalize(dct)) if render else dct

    @staticmethod
    def _wordcount(X, max_words: int = None, exclude_words: list = []):
        wordcount = pd\
            .Series(X, dtype=object)\
            .apply(lambda x: x.split() if type(x) == str else x)\
            .explode()\
            .value_counts()[:max_words]\
            .drop(exclude_words, errors="ignore")
        wordcount.index.name = "index"
        wordcount.name = "value"
        return wordcount

    @staticmethod
    def _render(dct: dict):
        with open(D3HTML, 'r') as f:
            d3html = f.read()

        if isfile(D3JS):
            with open(D3JS, 'r') as f:
                d3js = f.read()
        else:
            d3js = urlopen(URL).read().decode("utf8")
            with open(D3JS, "w") as f:
                f.write(d3js)

        return str(d3html % (d3js, json.dumps(dct, indent=2)))

    @staticmethod
    def __normalize(dct: dict):
        count_range = (max(dct.values()) - min(dct.values()) + 1) if dct else 1
        size_ratio = 100.0 / count_range
        return [{
            'text': key,
            'size': int(value*size_ratio)+10,
        } for key, value in dct.items()]
