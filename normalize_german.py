#pip install pandas
#pip install spacy
#python -m spacy download de_core_news_sm

import pandas as pd
import spacy
nlp = spacy.load('de_core_news_sm')

def normalize_german(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)


def process_german_keywords(csv_file: str) -> list:
    df = pd.read_csv(csv_file)
    normalized_keywords = []
    for keyword in df['German']:
        normalized = normalize_german(keyword)
        normalized_keywords.append(normalized)
    return normalized_keywords