import spacy
import gensim
from gensim import corpora
from spacy.lang.de.stop_words import STOP_WORDS
from string import punctuation
import re
import pandas as pd

# Load the German language model
nlp = spacy.load('de_core_news_sm')

# Function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ''.join([char for char in text if char not in punctuation])  # Remove punctuation

    # Process with spaCy
    doc = nlp(text)

    # Tokenize, remove stop words, and lemmatize
    tokens = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS and not token.is_punct and not token.is_space]
    
    return tokens

# Loads the dataset
ds = pd.read_csv("german.csv")
corpus = ds["text"]

# Preprocess all the documents in the corpus
processed_corpus = [preprocess_text(doc) for doc in corpus]

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(processed_corpus)
corpus_gensim = [dictionary.doc2bow(text) for text in processed_corpus]

# Build the LDA model
lda_model = gensim.models.LdaMulticore(corpus_gensim, num_topics=3, id2word=dictionary, passes=10, workers=2)

# Print the topics
for idx, topic in lda_model.print_topics(num_topics=3, num_words=5):
    print(f"Topic {idx}: {topic}")

# Optionally, get the topic distribution for each document
for i, doc in enumerate(corpus_gensim):
    print(f"Document {i} topic distribution: {lda_model.get_document_topics(doc)}")
