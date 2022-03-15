#!/usr/bin/env python3

"""
Based on the Python code from twarc:
* https://github.com/DocNow/twarc/

Original d3.js wordcloud code by Jason Davies:
* http://github.com/jasondavies/d3-cloud
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import basename, dirname, isdir, isfile, splitext

from sklearn.pipeline import Pipeline
from typing import Callable, Union

import base.stopwords as stopwords
from base.base import PandasTransformer
from base.lemmatizer import Lemmatizer
from base.ngrams import NGrams
from base.stemmer import Stemmer
from base.tokenizer import Tokenizer
from base.wordcloud import Wordcloud

ENCODING = "utf-8"
IGNORE_STARTSWITH = ["http", "www", "kk"]
IGNORE_STARTSWITH_CHARS = "@#"
MAX_WORDS = 100
MIN_WORD_LEN = 2
N_GRAMS = 1

AVAILABLE_STOPWORDS = [
    "all",
    "catalan",
    "chinese",
    "common",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "portuguese",
    "russian",
    "spanish",
]

class WordcloudNLP(Pipeline):

    def __init__(
        self,
        applymap: Callable = lambda x: x,
        column: Union[str, list] = None,
        drop_duplicates: bool = False,
        dropna: bool = False,
        exclude_words: list = [],
        ignore_startswith: list = IGNORE_STARTSWITH,
        ignore_startswith_chars: str = IGNORE_STARTSWITH_CHARS,
        ignore_stopwords: bool = True,
        json_records: bool = True,
        lang: str = None,
        low_memory: bool = False,
        max_words: int = MAX_WORDS,
        min_word_len: int = MIN_WORD_LEN,
        model: str = None,
        n_grams: int = N_GRAMS,
        sep: str = None,
        skiprows: int = None,
        sort: list = [],
        stop_words: Union[str, list] = [],
        use_lemmas: bool = False,
        use_pandas: bool = False,
        use_stemmer: bool = False,
        use_tokens: bool = False,
        use_wordcloud: bool = True,
    ):
        steps = []

        self.applymap = applymap
        self.column = column
        self.drop_duplicates = drop_duplicates
        self.dropna = dropna
        self.exclude_words = exclude_words
        self.ignore_stopwords = ignore_stopwords
        self.ignore_startswith = ignore_startswith
        self.ignore_startswith_chars = ignore_startswith_chars
        self.json_records = json_records
        self.lang = lang
        self.low_memory = low_memory
        self.max_words = max_words
        self.min_word_len = min_word_len
        self.model = model
        self.n_grams = n_grams
        self.sep = sep
        self.skiprows = skiprows
        self.sort = sort
        self.stop_words = stop_words
        self.use_lemmas = use_lemmas
        self.use_pandas= use_pandas
        self.use_stemmer = use_stemmer
        self.use_tokens = use_tokens
        self.use_wordcloud = use_wordcloud

        if self.use_pandas:
            steps.append(
                ('pandas', PandasTransformer(
                    applymap=self.applymap,
                    column=self.column,
                    drop_duplicates=self.drop_duplicates,
                    dropna=self.dropna,
                    json_records=self.json_records,
                    low_memory=self.low_memory,
                    sep=self.sep,
                    skiprows=self.skiprows,
                    sort=self.sort,
                ))
            )
        if self.use_tokens:
            steps.append(
                ('token', Tokenizer(
                    ignore_startswith=self.ignore_startswith,
                    min_word_len=self.min_word_len,
                    stop_words=self.__stopwords(self.stop_words),
                ))
            )
        if self.use_lemmas:
            steps.append(
                ('lemma', Lemmatizer(
                    model=self.model,
                ))
            )
        if self.use_stemmer:
            steps.append(
                ('stem', Stemmer(
                    ignore_startswith=self.ignore_startswith_chars,
                    ignore_stopwords=self.ignore_stopwords,
                    lang=self.lang,
                ))
            )
        if self.n_grams:
            steps.append(
                ('ngrams', NGrams(
                    n_grams=self.n_grams,
                ))
            )
        if self.use_wordcloud:
            steps.append(
                ('wordcloud', Wordcloud(
                    exclude_words=self.exclude_words,
                    max_words=self.max_words,
                ))
            )
        super().__init__(steps=steps)

    @staticmethod
    def __stopwords(s):
        return getattr(stopwords, f"{s.upper()}_STOPWORDS") if s and type(s) == str else []


def getargs():
    argparser = ArgumentParser()

    argparser.add_argument("input",
                           help="Input file names or folder",
                           nargs="+")

    argparser.add_argument("-o", "--output-name",
                           dest="output",
                           help=f"Output file name and/or path")

    argparser.add_argument("-n", "--n-grams",
                           help=f"Length of n-grams (default: {N_GRAMS})",
                           default=N_GRAMS,
                           type=int)

    argparser.add_argument("-w", "--max-words",
                           help=f"Maximum words in cloud (default: {MAX_WORDS})",
                           default=MAX_WORDS,
                           type=int)

    argparser.add_argument("-x", "--exclude-words",
                           default=[],
                           help=f"Extra words to ignore for word cloud (comma separated)",
                           type=lambda x: x.split(","))

    argparser.add_argument("--column",
                           help=f"Column names or positions (comma separated)",
                           type=lambda x: x.split(","))

    argparser.add_argument("--delimiter",
                           dest="sep",
                           help=f"Character delimiter to load file")

    argparser.add_argument("--ignore_startswith-chars",
                           help=f"Characters to ignore for stemmer (default: {IGNORE_STARTSWITH_CHARS})",
                           default=IGNORE_STARTSWITH_CHARS)

    argparser.add_argument("--ignore-startswith",
                           help=f"Strings to ignore for tokenizer (comma separated; default: {IGNORE_STARTSWITH})",
                           default=IGNORE_STARTSWITH,
                           type=lambda x: x.split(","))

    argparser.add_argument("--lang-stemmer",
                           dest="lang",
                           help=f"Language to use for NLTK SnowBall stemmer (optional)")

    argparser.add_argument("--lang-stopwords",
                           default="all",
                           dest="stop_words",
                           help=f"Stopwords to use for tokenizer (default: all; available: {AVAILABLE_STOPWORDS})")

    argparser.add_argument("--min-word-len",
                           default=MIN_WORD_LEN,
                           help=f"Minimum word length for tokenizer (default: {MIN_WORD_LEN})",
                           type=int)

    argparser.add_argument("--model-spacy",
                           dest="model",
                           help=f"spaCy model to use (required for lemmatizer)")

    argparser.add_argument("--skiprows",
                           help=f"Number of rows to skip for Pandas",
                           type=int)

    argparser.add_argument("--no-pandas",
                           action="store_false",
                           dest="use_pandas",
                           help="Do NOT use pandas in pipeline")

    argparser.add_argument("--no-stopwords",
                           action="store_const",
                           const=[],
                           dest="stop_words",
                           help=f"Do NOT use any stopwords for tokenizer")

    argparser.add_argument("--no-tokens",
                           action="store_false",
                           dest="use_tokens",
                           help="Do NOT use tokenizer in pipeline")

    argparser.add_argument("--use-lemmas",
                           action="store_true",
                           help="Use lemmatizer in pipeline")

    argparser.add_argument("--use-stemmer",
                           action="store_true",
                           help="Use stemmer in pipeline")

    args = argparser.parse_args()
    return vars(args)


def getfiles(lst):
    files = []
    for name in (lst if type(lst) == list else [lst]):
        if isdir(name):
            [files.append(f) for f in sorted([f"{name}/{f}" for f in listdir(name)]) if isfile(f)]
        else:
            files.append(name)
    return files


def main(**args):
    files = getfiles(args.pop("input"))

    output = args.pop("output")
    output_file = basename(output) if output else ("%s_wordcloud" % splitext(basename(files[0]))[0])
    output_folder = dirname(output if output else ".") or "."

    if not isdir(output_folder):
        mkdir(output_folder)

    nlp = WordcloudNLP(**args)
    wordcloud = nlp.steps.pop(-1)[1]

    wordcount = wordcloud._wordcount(
        nlp.transform(files),
        exclude_words=wordcloud.exclude_words
    )
    wordcount.to_excel(f"{output_folder}/{output_file}.xlsx")

    with open(f"{output_folder}/{output_file}.html", "w") as f:
        f.write(
            wordcloud._wordcloud(wordcount[:wordcloud.max_words].to_dict())
        )


if __name__ == "__main__":
    main(**getargs())
