import re
import string

from .base import Transformer

ACCENT_REPLACEMENTS = {
    ord("á"): "a", ord("ã"): "a", ord("â"): "a",
    ord("à"): "a", ord("è"): "e", ord("ê"): "e",
    ord("é"): "e", ord("í"): "i", ord("ì"): "i",
    ord("ñ"): "n", ord("ò"): "o", ord("ó"): "o",
    ord("ô"): "o", ord("õ"): "o", ord("ù"): "u",
    ord("ú"): "u", ord("ü"): "u", ord("ç"): "c"}

VALID_CHARACTERS = "@#"
INVALID_CHARACTERS = "\\\"'’…|–—“”‘„•¿¡"

CHARACTER_REPLACEMENTS = str.maketrans("", "", "".join(
    set(string.punctuation + INVALID_CHARACTERS) - set(VALID_CHARACTERS)))

class Tokenizer(Transformer):

    def __init__(
        self,
        ignore_startswith: list = [],
        min_word_len: int = 0,
        stop_words: list = [],
    ):
        self.ignore_startswith = ignore_startswith
        self.min_word_len = min_word_len
        self.stop_words = stop_words

    def transform(self, X):
        return [
            "\n".join(
                " ".join([
                    w
                    .replace("](", " ")  # Markdown
                    .translate(ACCENT_REPLACEMENTS)
                    .translate(CHARACTER_REPLACEMENTS)
                    for w in
                        self._clear_emojis(sent)
                        .lower()
                        .split()
                    if
                        len(w) >= self.min_word_len
                    and
                        not self._is_number(w)
                    and
                        not any(w.startswith(_) for _ in self.ignore_startswith)
                    and
                        w.strip(VALID_CHARACTERS) not in self.stop_words
                ])
                for sent in (
                    x.split("\n")
                )
            )
            for x in X
        ]

    @staticmethod
    def _is_number(str_word):
        try:
            int(str_word)
        except:
            try:
                float(str_word)
            except:
                return False
        return True

    @staticmethod
    def _clear_emojis(str_text, replace_with=r' '):
        return re\
            .compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"  # extra (1)
                u"\U000024C2-\U0001F251"  # extra (2)
                u"\U0000200B-\U0000200D"  # zero width
                "]+", flags=re.UNICODE)\
            .sub(replace_with, str_text)