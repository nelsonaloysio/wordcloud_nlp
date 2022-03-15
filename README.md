# wordcloud-nlp

Natural language processor and word cloud generator in Python 3.

### Requirements

* langdetect (>=1.0.9)
* nltk (>=3.6.2)
* pandas (>=1.4.1)
* scikit-learn (>=0.24.2)

### Usage

```
usage: wordcloud_nlp [-h] [-o OUTPUT] [-n N_GRAMS] [-w MAX_WORDS]
                     [-x EXCLUDE_WORDS] [--column COLUMN] [--delimiter SEP]
                     [--ignore_startswith-chars IGNORE_STARTSWITH_CHARS]
                     [--ignore-startswith IGNORE_STARTSWITH]
                     [--lang-stemmer LANG] [--lang-stopwords STOP_WORDS]
                     [--min-word-len MIN_WORD_LEN] [--model-spacy MODEL]
                     [--skiprows SKIPROWS] [--no-pandas] [--no-stopwords]
                     [--no-tokens] [--use-lemmas] [--use-stemmer]
                     input [input ...]

positional arguments:
  input                 Input file names or folder

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output-name OUTPUT
                        Output file name and/or path
  -n N_GRAMS, --n-grams N_GRAMS
                        Length of n-grams (default: 1)
  -w MAX_WORDS, --max-words MAX_WORDS
                        Maximum words in cloud (default: 100)
  -x EXCLUDE_WORDS, --exclude-words EXCLUDE_WORDS
                        Extra words to ignore for word cloud (comma separated)
  --column COLUMN       Column names or positions (comma separated)
  --delimiter SEP       Character delimiter to load file
  --ignore_startswith-chars IGNORE_STARTSWITH_CHARS
                        Characters to ignore for stemmer (default: @#)
  --ignore-startswith IGNORE_STARTSWITH
                        Strings to ignore for tokenizer (comma separated;
                        default: ['http', 'www', 'kk'])
  --lang-stemmer LANG   Language to use for NLTK SnowBall stemmer (optional)
  --lang-stopwords STOP_WORDS
                        Stopwords to use for tokenizer (default: all;
                        available: ['all', 'catalan', 'chinese', 'common',
                        'english', 'french', 'german', 'italian', 'japanese',
                        'portuguese', 'russian', 'spanish'])
  --min-word-len MIN_WORD_LEN
                        Minimum word length for tokenizer (default: 2)
  --model-spacy MODEL   spaCy model to use (required for lemmatizer)
  --skiprows SKIPROWS   Number of rows to skip for Pandas
  --no-pandas           Do NOT use pandas in pipeline
  --no-stopwords        Do NOT use any stopwords for tokenizer
  --no-tokens           Do NOT use tokenizer in pipeline
  --use-lemmas          Use lemmatizer in pipeline
  --use-stemmer         Use stemmer in pipeline
```
