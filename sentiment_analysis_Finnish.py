from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

pipe = pipeline("text-classification", model="nisancoskun/bert-finnish-sentiment-analysis-v2")
tokenizer = AutoTokenizer.from_pretrained("nisancoskun/bert-finnish-sentiment-analysis-v2")
model = AutoModelForSequenceClassification.from_pretrained("nisancoskun/bert-finnish-sentiment-analysis-v2")

def analyze_senti_finnish_one_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    predicted_sentiment = torch.argmax(probs).item()
    if predicted_sentiment == 1:
        return "positive"
    return "negative"

def analyze_senti_finnish(csvfile):
    df = pd.read_csv(csvfile)
    df["sentiment"] = df["text"].apply(analyze_senti_finnish_one_text)
    return df
    df.to_csv("sentiment_analysis_fi.csv", index=False)

