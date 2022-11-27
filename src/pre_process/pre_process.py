# %%
import pandas as pd
import utils
from collections import namedtuple
import spacy
import process_ngrams
import spacy


replace_phrase_info = namedtuple("replace_phrase_info", ["phrase_to_match", "replacement_phrase"])

replace_phrase_scenarios = [
    replace_phrase_info("Ruby Tuesday", "BurgerPalace"),
    replace_phrase_info("Ruby", "BurgerPalace"),
]

def run():
    nlp = spacy.load("en_core_web_sm")

    review_df = pd.read_csv("rt_reviews.csv")
    review_df = review_df[review_df["review_date"] > "2020-01-01"]

    for scenario in replace_phrase_scenarios:
        review_df = utils.replace_phrase(review_df, scenario.phrase_to_match, scenario.replacement_phrase)


    processed_text = {}
    text_df = pd.DataFrame()

    for i, row in review_df.iterrows():
        doc = nlp(row["review_text"])
        processed_text[row["review_id"]] = utils.ProcessedText(
            id=row["review_id"],
            date=row["review_date"],
            text_source="yelp",
            text=doc.text,
            rating=row["review_rating"],
            norm_text=utils.normalize_text(doc),
            word_tokens=utils.get_tokens(doc),
            sentence_tokens=utils.get_sentence_tokens(doc),
        )
        text_df = pd.concat([pd.DataFrame({
            "review_id": row["review_id"],
            "word_count": processed_text[row["review_id"]].word_count,
            "sentence_count": processed_text[row["review_id"]].sentence_count,
            "rating_category": processed_text[row["review_id"]].rating_category,
            "norm_text": processed_text[row["review_id"]].norm_text,
            "processed_text": processed_text[row["review_id"]],
        }, index=[0]), text_df])

    text_df = review_df.merge(text_df, on="review_id")
    text_df["review_month_year"] = pd.to_datetime(text_df["review_date"]) + pd.offsets.MonthBegin(1)
    text_df["review_text_lower"] = text_df["review_text"].str.lower() 
    text_df["review_sentences"] = text_df["review_text"].apply(lambda x: [str(sent) for sent in nlp(x).sents])
    text_df = process_ngrams.run(text_df)

    text_df.to_pickle("cleaned_review_text.pkl")
    return text_df

