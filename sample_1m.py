import warnings
warnings.filterwarnings("ignore", module="urllib3")

import io
import json
import os
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import nltk
import regex as re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

LANGUAGE = "zho_Hans"
SAMPLE_FILE = Path(f"{LANGUAGE}.shuf")
N_RESULTS = 200

def read_json_lines(path: Path):
    with path.open("r") as f:
        for line in f:
            yield json.loads(line)

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

_NORMALIZERS = {
    "eng_Latn": _normalize_text_eng,
    "zho_Hans": _normalize_text_zho_hans,
    "ben_Beng": _normalize_text_ben
}

def normalize_text(text):
    return _NORMALIZERS[LANGUAGE](text)

def read_keywords(path: Path):
    with path.open("r") as f:
        return [normalize_text(kw.strip()) for kw in f.readlines() if not kw.startswith("#")]
    
KEYWORDS = read_keywords(Path(f"keywords.{LANGUAGE}.txt"))

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

    documents = read_json_lines(SAMPLE_FILE)

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

    df.to_csv(f"samples-{LANGUAGE}-1m-200.csv")