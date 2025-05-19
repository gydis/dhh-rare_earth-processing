from cemotion import Cemotion
from cnsenti import Sentiment
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

senti = Sentiment() # for formal texts
c = Cemotion() # for informal texts

def analyze_senti_formal_ch(text):
    senti_dict = senti.sentiment_calculate(text) # this is really slow - maybe we should parallelise
    pos = senti_dict['pos']
    neg = senti_dict['neg']
    if pos > neg:
        return 'positive'
    elif neg > pos:
        return 'negative'
    else:
        return 'neutral'
    # or if we want: 
    # return senti_dict
    # It's a dict of scores e.g. {'sentences': 2, 'words': 22, 'pos': np.float64(27.0), 'neg': np.float64(0.0)}

def analyze_senti_informal_ch(text):
    # cemotion score is sentiment polarity with 0 being negative and 1 being positive
    score = c.predict(text)
    return score

def analyze_sentiment_chinese(csvfile, formality):
    df = pd.read_csv(csvfile)
    if formality == "formal":
        df["sentiment / formal"] = df["text"].apply(analyze_senti_formal_ch)
        df.to_csv("formal_sentiment_analysis_ch.csv", index=False)
    elif formality == "informal":
        df["sentiment / informal"] = df["text"].apply(analyze_senti_informal_ch)
        df.to_csv("informal_sentiment_analysis_ch.csv", index=False)