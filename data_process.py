import datasets
from joblib import Parallel, delayed
import itertools as it
from tqdm import tqdm
import sys
import pandas as pd

#all lower case
#get rid of punctuation?
#lemmatize the keywords
#input is a dictionary ("text": )
#write one function for each language
#comparing function extra?
#output: true or false (edited) 

lang = "rus_Cyrl"
BATCH_SIZE = 1000

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

keywords_df = pd.read_csv('Keywords - Лист1.csv')  

def normalize_keywords(lang, keywords):
    language = lang_dict[lang]
    lang_keywords = keywords_df[language].dropna()
    normalized_keywords = [lang_func_norm_dummy(word) for word in lang_keywords]
    return normalized_keywords

dict_of_normalized_keyword_lists = {}

for l in lang_dict:
    normalized_keywords = normaliz_keywords(l, keywords_df)
    dict_of_normalized_keyword_lists[l] = normalized_keywords

def compare_funct(text, keywords):
    for word in keywords:
        if word in text:
            return True
    return False

def process_chunk(chunk_index: int, chunk: tuple):



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

parallel = Parallel(n_jobs=-1) # TODO: config backend etc

# Launch the processing
parallel(delayed(process_chunk)(i, batch) for i, batch in tqdm(batched(iterator, BATCH_SIZE)))