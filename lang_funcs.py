import spacy

nlp_ru = spacy.load("ru_core_news_sm", disable=["parser", "ner"])

def rus_normalize(text):
    text = text.lower()
    # get rid of punctuation
    text.replace('\n', ' ')
    print(text)
    text = ''.join(list(filter(lambda x: x.isalpha() or x == ' ' or x == '-', text)))
    doc = nlp_ru(text)
    normed_text = ''.join([token.lemma_ for token in doc])
    return normed_text