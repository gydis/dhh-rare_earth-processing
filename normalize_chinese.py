#pip3 install pkuseg

import pkuseg
import pandas as pd

def normalize_chinese(text:str):
    seg = pkuseg.pkuseg()         
    stems = seg.cut(text)  
    return " ".join(stems)

def process_chinese_keywords(csv_file: str) -> list:
    df = pd.read_csv(csv_file)
    normalized_keywords = []
    for keyword in df['Chinese']:
        normalized = normalize_chinese(keyword)
        normalized_keywords.append(normalized)
    return normalized_keywords