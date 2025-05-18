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
import argparse
from lang_funcs import lang_to_func
import time

#all lower case
#get rid of punctuation?
#lemmatize the keywords
#input is a dictionary ("text": )
#write one function for each language
#comparing function extra?
#output: true or false (edited) 

# TODO: add command line arguments for language, batch size and proportion of data

parser = argparse.ArgumentParser(description='Apply normalization and filtering to dataset.')
parser.add_argument('--lang', type=str, help='Language code (e.g., rus_Cyrl, eng_Latn) from the hplt dataset')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for processing into one parquet file')
parser.add_argument('--chunk_size', type=int, help='Chunk size for processing on the node')
parser.add_argument('--chunk_ind', type=int, help='Index of the chunk to process')

args = parser.parse_args()

lang_sizes = {
    "rus_Cyrl": 884688865,
    "eng_Latn": 4388525961,
    "fin_Latn": 34815601,
    "zho_Hans": 1403640133,
    "deu_Latn": 482053407,
    "ben_Beng": 11043918,
    "hin_Deva": 13651945
}

lang = args.lang
if lang not in ["rus_Cyrl", "eng_Latn", "fin_Latn", "zho_Hans", "deu_Latn", "ben_Beng", "hin_Deva"]:
    print(f"Language {lang} is not supported. Please choose from the following: rus_Cyrl, eng_Latn, fin_Latn, zho_Hans, deu_Latn, ben_Beng, hin_Deva.")
    sys.exit(1)

BATCH_SIZE = args.batch_size


chunk_size = args.chunk_size
if chunk_size is None:
    print("Chunk size not provided")
    sys.exit(1)

chunk_ind = args.chunk_ind
if args.chunk_ind is None:
    print("Chunk index not provided")
    sys.exit(1)

lang = "rus_Cyrl" # TODO remove
chunk_ind = 0 # TODO remove
chunk_size = 100000 # TODO remove

PATH_OUT = "data/processed/"
os.makedirs(PATH_OUT, exist_ok=True)

lang_to_keycol = {
    "rus_Cyrl": "Russian",
    "eng_Latn" : "English",
    "fin_Latn" : "Finnish",
    "zho_Hans" : "Chinese",
    "deu_Latn" : "German",
    "ben_Beng" : "Bangla",
    "hin_Deva" : "Hindi"
            }

lang_func_norm = lang_to_func(lang)

# Read the key csv
# Normalize the keys in the list

keywords_df = pd.read_csv('keywords.csv')  

lang_keywords = keywords_df[lang_to_keycol[lang]].dropna()
keywords = [lang_func_norm(word) for word in lang_keywords]

print("Keywords loaded and normalized")

def compare_funct(text):
    text = lang_func_norm(text)
    split_text = text.split(" ")
    sets = {}
    for word in keywords:
        words = word.split(" ")
        w_len = len(words)
        if w_len not in sets.keys():
            sets[w_len] = set([" ".join(split_text[i:i+w_len]) for i in range(len(split_text)-w_len+1)])
        # if word in sets[w_len]:
        #     return True
        if re.search(rf'/b{re.escape(word)}/b', sets[w_len]):
            return True
    return False

print(f"Loading dataset, {time.strftime("%H:%M:%S", time.localtime())}")
dataset = datasets.load_dataset("HPLT/HPLT2.0_cleaned", lang, split='train', streaming=True)
dataset = dataset.skip(chunk_ind * chunk_size).take(chunk_size)
dataset_batched = dataset.batch(BATCH_SIZE)
print(f"Dataset Loaded, {time.strftime("%H:%M:%S", time.localtime())}")

def process_batch(batch_index: int, batch: dict) -> int:
    # Process batch
    texts = batch["text"]
    mask = [compare_funct(text) for text in texts]
    for key in batch.keys():
        batch[key] = [batch[key][i] for i in range(len(batch[key])) if mask[i]]
    table = pa.Table.from_pydict(batch)
    pq.write_table(table, f"{PATH_OUT}{chunk_ind}_{batch_index}.parquet")
    return batch_index

iterator = iter(dataset_batched)

parallel = Parallel(n_jobs=-1) # TODO: config backend

# Launch the processing
print(f"Processing batches, {time.strftime("%H:%M:%S", time.localtime())}")
parallel(delayed(process_batch)(i, batch) for i, batch in tqdm(enumerate(iterator)))
print(f"Processing finished, {time.strftime("%H:%M:%S", time.localtime())}")