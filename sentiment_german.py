#pip install spacy textblob spacytextblob
#python -m spacy download de_core_news_sm


from textblob import TextBlob
import spacy
import pandas as pd

# Load the German language model
nlp = spacy.load("de_core_news_sm")

# Function for sentiment analysis in German using TextBlob
def sentiment_analysis(text):
    # Translate text to English for better sentiment analysis accuracy
    blob = TextBlob(text)
    
    # Return the sentiment polarity (-1 = negative, 1 = positive)
    return blob.sentiment.polarity

# Example German sentences for sentiment analysis
# Loads the dataset
ds = pd.read_csv("german.csv")
german_texts = ds["text"]


for text in german_texts:
    sentiment = sentiment_analysis(text)
    print(f"Text: {text}\nSentiment Polarity: {sentiment}\n")
