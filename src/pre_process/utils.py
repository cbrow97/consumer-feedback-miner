from difflib import SequenceMatcher
import pandas as pd
from dataclasses import dataclass
import datetime

def replace_phrase(text_df: pd.DataFrame, phrase_to_match: str, replacement_phrase: str) -> pd.DataFrame:
    len_of_phrase = len(phrase_to_match.split(" "))

    for df_i, row in text_df.iterrows():
        review_text = row["review_text"].split(" ")

        for i, __ in enumerate(review_text):
            text_to_assess = " ".join(review_text[i:i+len_of_phrase])
            ratio = SequenceMatcher(None, phrase_to_match, text_to_assess).ratio()
            
            if ratio >= .85:
                text_df.loc[df_i, "review_text"] = text_df.loc[df_i, "review_text"].replace(text_to_assess, replacement_phrase)

    return text_df


def get_stop_words(doc):
    return [t for t in doc if t.is_stop]


def get_lemmas(doc):
    return [t.lemma_ for t in doc]


def get_punctuations(doc):
    return [t for t in doc if t.pos_ == "PUNCT"]


def get_tokens(doc):
    return [str(t) for t in doc if t not in get_punctuations(doc)]


def get_sentence_tokens(doc):
    return [[t.text for t in sent] for sent in doc.sents]


def get_sentences(doc):
    return [sent for sent in doc.sents]


def normalize_text(doc):
    return " ".join([
        str(t) for t in doc if 
            t not in get_stop_words(doc)
            and t not in get_punctuations(doc) 
            and str(t) in get_lemmas(doc) 
    ]).lower()

@dataclass
class ProcessedText:
    id: str
    date: datetime
    text_source: str
    text: str
    rating: int
    norm_text: str
    word_tokens: list
    sentence_tokens: list
    
    @property
    def word_count(self):
        return len(self.word_tokens)

    @property
    def sentence_count(self):
        return len(self.sentence_tokens)

    @property
    def rating_category(self):
        return {
            1: "negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "positive",
        }[self.rating]
