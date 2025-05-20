#pip install turftopic[jieba] datasets sentence_transformers topicwizard

from turftopic.vectorizers.chinese import ChineseCountVectorizer
from sentence_transformers import SentenceTransformer
from turftopic import KeyNMF
import urllib.request
import topicwizard
import pandas as pd
import numpy as np

# Loads the dataset
ds = pd.read_csv("samples-with-labels-zho_Hans-1m-200.csv")
corpus = ds["text"]

vectorizer = ChineseCountVectorizer(stop_words="chinese")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = KeyNMF(
    n_components=20,
    top_n=25,
    vectorizer=vectorizer,
    encoder=encoder,
    random_state=42, # Setting seed so that our results are reproducible
)

document_topic_matrix = model.fit_transform(corpus)
model.print_topics()

urllib.request.urlretrieve(
    "https://github.com/shangjingbo1226/ChineseWordCloud/raw/refs/heads/master/fonts/STFangSong.ttf",
    "./STFangSong.ttf",
)
topicwizard.visualize(
    corpus=corpus, model=model, wordcloud_font_path="./STFangSong.ttf"
)