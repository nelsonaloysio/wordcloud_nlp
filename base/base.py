from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import json
import pandas as pd
from sklearn.pipeline import Pipeline


class Transformer(metaclass=ABCMeta):
    """
    Asbtract base transformer class.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """ Abstract initializer class. """
        # super().__init__()

    def fit(self, X, y=None):
        """ Just returns class, nothing to fit. """
        return self

    @abstractmethod
    def transform(self, X):
        """ Abstract method for "DIY" transformations. """


class PandasTransformer(Transformer):

    def __init__(
        self,
        applymap: Callable = lambda x: x,
        column: Union[str, list] = None,
        drop_duplicates: bool = False,
        dropna: bool = False,
        json_records: bool = True,
        low_memory: bool = False,
        sep: str = None,
        skiprows: int = None,
        sort: list = [],
    ):
        self.applymap = applymap
        self.column = column
        self.drop_duplicates = drop_duplicates
        self.dropna = dropna
        self.json_records = json_records
        self.low_memory = low_memory
        self.sep = sep
        self.skiprows = skiprows
        self.sort = sort

    def transform(self, path_or_df: Union[str, pd.Series, pd.DataFrame]) -> pd.Series:
        series = self.__concat(
            [self.__read_json(
                x,
                json_records=self.json_records
            )
            if
                type(x) == str and x.endswith(".json")
            else
                self.__read_records(x)
            if
                type(x) == str and self.sep is None
            else
                pd.read_table(
                    x,
                    low_memory=self.low_memory,
                    sep=self.__get_file_delimiter(x) if self.sep is None and self.column else self.sep,
                    skiprows=self.skiprows,
                    usecols=list(set(
                        ((self.column if type(self.column) == list else [self.column]) if self.column is not None else []) +
                        ((self.sort if type(self.sort) == list else [self.sort]) if self.sort is not None else [])
                    )) or None,
                )
            if
                type(x) == str
            else
                x
            for x in
                (path_or_df if type(path_or_df) == list else [path_or_df])
            ],
            column=self.column,
        )

        if type(series) == pd.DataFrame:
            raise TypeError(f"Expected a Pandas Series or 1-dimensional DataFrame (column='{self.column}').")

        series = series.apply(self.applymap)

        if self.sort:
            series.sort_values(self.sort, ascending=False)
        if self.drop_duplicates:
            series.drop_duplicates(inplace=True)
        if self.dropna:
            series.dropna(inplace=True)

        self.index_ = series.index
        self.skiprows_ = self.index_.difference(series.index)
        return series

    @staticmethod
    def __concat(dfs: list, column=None) -> pd.DataFrame:
        if column:
            dfs = [df[c] for df in dfs for c in (column if type(column) == list else [column])]
        df = pd.concat(dfs)
        df.index = range(df.shape[0])
        return df

    @staticmethod
    def __get_file_delimiter(path: str) -> str:
        delimiters = ["|", "\t", ";", ","]
        with open(path, "rt") as f:
            header = f.readline()
        for i in delimiters:
            if i in header:
                return i
        return "\n"

    @staticmethod
    def __read_json(path: str, json_records=False) -> pd.DataFrame:
        if json_records:
            with open(path, "r") as j:
                return pd.DataFrame(
                    [json.loads(_) for _ in j.readlines()]
                )
        return pd.read_json(path)

    @staticmethod
    def __read_records(path: str) -> pd.Series:
        with open(path, "r") as f:
            return pd.Series(
                [_ for _ in [_.rstrip() for _ in f.readlines()] if _],
                dtype=object,
            )
