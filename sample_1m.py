import warnings
warnings.filterwarnings("ignore", module="urllib3")

import io
import json
import os
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Any

import nltk
import regex as re
import pandas as pd
import zstandard as zstd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import spacy
import string

LANGUAGE = "rus_Cyrl"
SAMPLE_FILE = Path(f"data/{LANGUAGE}.shuf.zst")
N_RESULTS = 2

def read_zstd_json_lines(path: Path, *, encoding = "utf-8") -> Iterable[Any]:
    """Return an iterator over the records in a Zstandard-compressed JSON Lines file"""
    with path.open("rb") as f:
        ctx = zstd.ZstdDecompressor()
        with ctx.stream_reader(f) as reader:
            textio = io.TextIOWrapper(reader, encoding=encoding)
            for line in textio:
                yield json.loads(line.rstrip("\n"))

@cache
def _cached_pkuseg():
    import pkuseg
    return pkuseg.pkuseg()

@cache
def _cached_bangla_stemmer():
    from bangla_stemmer.stemmer.stemmer import BanglaStemmer
    BanglaStemmer()

def _normalize_text_eng(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = (t for t in tokens if t.isalnum())
    return " ".join(WordNetLemmatizer().lemmatize(t) for t in tokens)

def _normalize_text_zho_hans(text):
    tokens = _cached_pkuseg().cut(text)
    return " ".join(t for t in tokens if t.isalnum())

def _normalize_text_ben(text):  
    from bnlp import NLTKTokenizer

    tokenizer = NLTKTokenizer()
    stemmer = _cached_bangla_stemmer()

    tokens = tokenizer.word_tokenize(text)
    return " ".join(stemmer.stem(t) for t in tokens)

nlp_ru = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
def _normalize_text_rus(text):
    text = text.lower()
    text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp_ru(text.lower())
    normed_text = ' '.join([token.lemma_ for token in doc])
    return normed_text

_NORMALIZERS = {
    "eng_Latn": _normalize_text_eng,
    "zho_Hans": _normalize_text_zho_hans,
    "ben_Beng": _normalize_text_ben,
    "rus_Cyrl": _normalize_text_rus,
}

_KEYWORD_COLS= {
    "rus_Cyrl": "Russian",
    "eng_Latn" : "English",
    "fin_Latn" : "Finnish",
    "zho_Hans" : "Chinese",
    "deu_Latn" : "German",
    "ben_Beng" : "Bangla",
    "hin_Deva" : "Hindi"
            }

def normalize_text(text):
    return _NORMALIZERS[LANGUAGE](text)

def read_keywords(path: Path):
    with path.open("r") as f:
        return [normalize_text(kw.strip()) for kw in f.readlines() if not kw.startswith("#")]
    
keywords_df = pd.read_csv('keywords.csv')  

lang_keywords = keywords_df[_KEYWORD_COLS[LANGUAGE]].dropna()

KEYWORDS = []
ngram_lens = set()
for keyword in lang_keywords:
    if '&' in keyword:
        keywords = keyword.split('&')
        for k in keywords:
            ngram_lens.add(len(k.split(" ")))
        keywords = list(map(lambda k: normalize_text(k), keywords))
        KEYWORDS.append('&'.join(keywords))
    else:
        ngram_lens.add(len(keyword.split(" ")))
        KEYWORDS.append(normalize_text(keyword))

def keyword_search_set(doc):
    text = normalize_text(doc["text"])
    split_text = text.split(" ")
    ngram_sets = {}
    for ngram_len in ngram_lens:
        ngram_sets[ngram_len] = set([" ".join(split_text[i:i+ngram_len]) for i in range(len(split_text)-ngram_len+1)])
    for keyword in KEYWORDS:
        if '&' in keyword:
            keywords = keyword.split('&')
            res = True
            for k in keywords:
                w_len = len(k.split(" "))
                if not k in ngram_sets[w_len]:
                    res = res and False
            return res
        else:
            w_len = len(keyword.split(" "))
            if keyword in ngram_sets[w_len]:
                return True
    return False

# KEYWORDS = read_keywords(Path(f"keywords.{LANGUAGE}.txt"))

PATTERN = re.compile(rf"[^\w+]({'|'.join(KEYWORDS)})[^\w+]")

def keyword_search(doc):
    text = normalize_text(doc["text"])
    return bool(PATTERN.search(text))

def scan(doc):
    if doc["u"].startswith("https://en.wikipedia.org/wiki/"):
        return None
    return doc if keyword_search(doc) else None

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    documents = read_zstd_json_lines(SAMPLE_FILE)

    matches = []
    search_count = 0
    with Pool(os.cpu_count() - 1) as pool, tqdm(total=N_RESULTS, desc="Keyword search") as pbar:
        for result in pool.imap_unordered(scan, documents, chunksize=10):
            search_count += 1
            if result is None:
                continue
            matches.append(result)
            pbar.set_postfix(n=search_count)
            pbar.update()
            if len(matches) >= N_RESULTS:
                pool.terminate()
                break

    print(f"Total {search_count} documents scanned for {N_RESULTS} matches (~{(N_RESULTS/search_count):.2%})")

    df = pd.DataFrame(matches)
    df = df.drop(columns=['f', 'o', 's', 'rs', 'c', 'collection', 'lang', 'prob', 'seg_langs', 'robotstxt', 'filter', 'pii', 'doc_scores'])

    df.to_csv(f"samples-{LANGUAGE}-combine_test.csv")