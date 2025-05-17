import datasets
from joblib import Parallel, delayed
import itertools as it
from tqdm import tqdm
import sys
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random
import re

#all lower case
#get rid of punctuation?
#lemmatize the keywords
#input is a dictionary ("text": )
#write one function for each language
#comparing function extra?
#output: true or false (edited) 

# TODO: add command line arguments for language, batch size and proportion of data
lang = "rus_Cyrl"

BATCH_SIZE = 10000
PATH_OUT = "data/processed/"
os.makedirs(PATH_OUT, exist_ok=True)

lang_dict = {
    "rus_Cyrl": "Russian",
    "eng_Latn" : "English",
    "fin_Latn" : "Finnish",
    "zho_Hans" : "Chinese",
    "deu_Latn" : "German",
    "ben_Beng" : "Bangla",
    "hin_Deva" : "Hindi"
            }

# The slice from language partition as a parameter

# lang_tofunc_dict?
def lang_func_norm_dummy(text):
    # Lemmatization etc
    normed_text = text.lower()
    return normed_text

# Read the key csv
# Normalize the keys in the list

keywords_df = pd.read_csv('keywords.csv')  

def normalize_keywords(lang, keywords):
    language = lang_dict[lang]
    lang_keywords = keywords_df[language].dropna()
    normalized_keywords = [lang_func_norm_dummy(word) for word in lang_keywords]
    return normalized_keywords

dict_of_normalized_keyword_lists = {}

for l in lang_dict:
    normalized_keywords = normalize_keywords(l, keywords_df)
    dict_of_normalized_keyword_lists[l] = normalized_keywords

keywords = dict_of_normalized_keyword_lists[lang]
def compare_funct(text):
    for word in keywords:
        if re.search(rf'/b{re.escape(word)}/b', text):
            return True
    return False

def process_chunk(chunk_index: int, chunk: tuple):
    # Process chunk
    data = list(filter(lambda item: compare_funct(lang_func_norm_dummy(item['text'])), chunk)) # TODO: make it more efficient?
    table = pa.Table.from_pylist(data)
    pq.write_table(table, f"{PATH_OUT}{chunk_index}.parquet")
    return chunk_index


dataset = datasets.load_dataset("HPLT/HPLT2.0_cleaned", lang, split='train', streaming=True)

def batched(iterator, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    i = -1
    while (batch := tuple(it.islice(iterator, n))):
        i += 1
        yield i, batch

iterator = it.islice(iter(dataset), 10000)

parallel = Parallel(n_jobs=-1) # TODO: config backend

# Launch the processing
parallel(delayed(process_chunk)(i, batch) for i, batch in tqdm(batched(iterator, BATCH_SIZE)))