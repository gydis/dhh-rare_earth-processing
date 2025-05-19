def lang_to_func(lang):
    match lang:
        case "rus_Cyrl":
            return rus_normalize
        case "eng_Latn":
            raise ValueError("English normalization is not implemented yet.")
        # case "fin_Latn":
        #     return normalize_finnish
        # case "zho_Hans":
        #     return normalize_chinese
        # case "deu_Latn":
        #     return normalize_german
        # case "ben_Beng":
        #     return normalize_bengali
        case "hin_Deva":
            return ValueError("Hindi normalization is not implemented yet.")
        case _:
            raise ValueError(f"Language {lang} is not supported.")

import spacy
import string
nlp_ru = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
def rus_normalize(text):
    text = text.lower()
    text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp_ru(text.lower())
    normed_text = ' '.join([token.lemma_ for token in doc])
    return normed_text

#pip install pandas
#pip install spacy
#python -m spacy download de_core_news_sm

# nlp_de = spacy.load('de_core_news_sm', disable=["parser", "ner"])

# def normalize_german(text):
#     text.replace('\n', ' ')
#     doc = nlp_de(text.lower())
#     lemmas = [token.lemma_ for token in doc]
#     return " ".join(lemmas)

# import pkuseg

# def normalize_chinese(text:str):
#     seg = pkuseg.pkuseg()         
#     stems = seg.cut(text)  
#     return " ".join(stems)


# # from bangla_stemmer.stemmer import stemmer
# # stmr = stemmer.BanglaStemmer()
# def normalize_bengali(text):
#     words = text.split()
#     stems = [ ]
#     for word in words:
#         stm = stmr.stem(word)
#         stems.append(stm)
#     return " ".join(stems)

# # Finnish:

# from trankit import Pipeline
# p = Pipeline(lang='finnish')

# def lemmatize_full_finnish_text(text):
#     text = text.lower()
#     text.replace('\n', ' ')
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     output = p.lemmatize(text, is_sent=True)
#     lemmatized_text = ' '.join(item['lemma'] for item in output['tokens'])
#     return lemmatized_text

# def lemmatize_finnish_keywords(keyword):
#     sentence = f'{keyword} on suomen kielen sana.'
#     lemmatized_sentence = lemmatize_full_finnish_text(sentence)
#     lemmatized_word = lemmatized_sentence.split()[0]
#     return lemmatized_word

# def normalize_finnish(text):
#     if (len(text.split()) < 2):
#         normalized_text = lemmatize_finnish_keywords(text)
#     else:
#         normalized_text = lemmatize_full_finnish_text(text)
#     return normalized_text

