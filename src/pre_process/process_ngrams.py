import nltk
from nltk.collocations import *
from nltk.corpus import stopwords
import pandas as pd
import string
import numpy as np

ngram_measures_dict = {
    "bigram": nltk.collocations.BigramAssocMeasures,
    "trigram": nltk.collocations.TrigramAssocMeasures,
    "quadgram": nltk.collocations.QuadgramAssocMeasures,
}

ngram_finder_dict = {
    "bigram": BigramCollocationFinder,
    "trigram": TrigramCollocationFinder,
    "quadgram": QuadgramCollocationFinder,
}

stops = set(stopwords.words('english'))

def get_ngram_freq_cutoff(finder):
    return (
        np.mean(list(finder.ngram_fd.values())) + 
        np.std(list(finder.ngram_fd.values())) 
        * 3
    )

def get_top_ngram(ngram: str, df: pd.DataFrame) -> pd.DataFrame:
    tokens = nltk.wordpunct_tokenize(''.join(df["review_text"].str.lower()))

    finder = ngram_finder_dict[ngram].from_words(tokens)

    finder.apply_word_filter(lambda w: w in string.punctuation)
    finder.apply_word_filter(lambda w: w in stops)
    finder.apply_word_filter(lambda w: w in ('I', 'me', "the"))

    ngram_freq_cutoff = get_ngram_freq_cutoff(finder)

    grams = [key for key, value in finder.ngram_fd.items() if value > ngram_freq_cutoff]

    return grams

def add_ngrams_field(text, grams):
    found_grams = []
    for gram in grams:
        if " ".join(gram) in text:
            found_grams.append(gram)

    return found_grams


def run(df: pd.DataFrame):
    ngrams = ["bigram", "trigram", "quadgram"]

    for ngram in ngrams:
        grams = get_top_ngram(ngram, df)
        df[ngram] = df["review_text_lower"].apply(lambda x: add_ngrams_field(x, grams))
        
    return df
