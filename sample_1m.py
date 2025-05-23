import warnings
warnings.filterwarnings("ignore", module="urllib3")

import io
import json
import os
import string
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

LANGUAGE = "rus_Cyrl"  # Change this to the desired language code
SAMPLE_FILE = Path(f"data/{LANGUAGE}.shuf.zst")
N_RESULTS = 200

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
    # import spacy_pkuseg as pkuseg
    return pkuseg.pkuseg()

@cache
def _cached_bangla_stemmer():
    from bangla_stemmer.stemmer.stemmer import BanglaStemmer
    BanglaStemmer()

@cache
def _cached_stanza_pipeline_fin():
    import stanza
    return stanza.Pipeline(lang="fi", processors="tokenize,mwt,lemma")

@cache
def _cached_spacy_pipeline_deu():
    import spacy
    return spacy.load("de_core_news_sm")

def _normalize_text_eng(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = (t for t in tokens if t.isalnum())
    return " ".join(WordNetLemmatizer().lemmatize(t) for t in tokens)

def _normalize_text_zho_hans(text):
    tokens = _cached_pkuseg().cut(text)
    return " ".join(t for t in tokens if t.isalnum())

def _normalize_text_fin(text):
    pipeline = _cached_stanza_pipeline_fin()
    doc = pipeline(text)
    tokens = (w.lemma for s in doc.sentences for w in s.words)
    # Lemmatized tokens might contain mwt separators ("#") so we need to reject tokens
    # based on exact comparison with punctuation characters instead of simple `t.isalnum()`
    return " ".join(t for t in tokens if t not in string.punctuation)

def _normalize_text_deu(text):
    pipeline = _cached_spacy_pipeline_deu()
    doc = pipeline(text.lower())
    return " ".join(token.lemma_ for token in doc)

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
    if len(text) > 1000000:
        text = text[:1000000]
    doc = nlp_ru(text.lower())
    normed_text = ' '.join([token.lemma_ for token in doc])
    return normed_text

_NORMALIZERS = {
    "eng_Latn": _normalize_text_eng,
    "fin_Latn": _normalize_text_fin,
    "zho_Hans": _normalize_text_zho_hans,
    "ben_Beng": _normalize_text_ben,
    "rus_Cyrl": _normalize_text_rus,
    "deu_Latn": _normalize_text_deu,
}

def normalize_text(text):
    return _NORMALIZERS[LANGUAGE](text)

def read_keywords(path: Path):
    with path.open("r") as f:
        return [normalize_text(kw.strip()) for kw in f.readlines() if not kw.startswith("#")]
    
lang_keywords = read_keywords(Path(f"keywords.{LANGUAGE}.txt"))

KEYWORDS = []
ngram_lens = set()
for keyword in lang_keywords:
    if '&' in keyword:
        keywords = keyword.split('&')
        keywords = list(map(normalize_text, keywords))
        for k in keywords:
            ngram_lens.add(len(k.split(" ")))
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
            assert len(keywords) > 0
            is_compound_match = True
            for k in keywords:
                w_len = len(k.split(" "))
                if k not in ngram_sets[w_len]:
                    is_compound_match = False
            if is_compound_match:
                return True
        else:
            w_len = len(keyword.split(" "))
            if keyword in ngram_sets[w_len]:
                return True
    return False

PATTERN = re.compile(rf"[^\w+]({'|'.join(KEYWORDS)})[^\w+]")

def keyword_search(doc):
    text = normalize_text(doc["text"])
    return bool(PATTERN.search(text))

def scan(doc):
    if doc["u"].startswith("https://en.wikipedia.org/wiki/"):
        return None
    return doc if keyword_search_set(doc) else None

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
            pbar.set_postfix(n=int(search_count))
            if result is None:
                continue
            matches.append(result)
            pbar.update()
            if len(matches) >= N_RESULTS:
                pool.terminate()
                break

    print(f"Total {search_count} documents scanned for {N_RESULTS} matches (~{(N_RESULTS/search_count):.2%})")

    df = pd.DataFrame(matches)
    df = df.drop(columns=['f', 'o', 's', 'rs', 'c', 'collection', 'lang', 'prob', 'seg_langs', 'robotstxt', 'filter', 'pii', 'doc_scores'])

    df.to_csv(f"samples-{LANGUAGE}-combined-v2.csv")