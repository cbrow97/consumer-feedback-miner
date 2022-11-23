# %%
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

class TextPreprocessor:
    def __init__(self, text: str, nlp):
        self.doc = nlp(text.lower())

    def get_stop_words(self):
        return [t for t in self.doc if t.is_stop]

    def get_lemmas(self):
        return [t.lemma_ for t in self.doc]

    def get_punctuations(self):
        return [t for t in self.doc if t.pos_ == "PUNCT"]

    def get_tokens(self, remove_punctuations=True):
        return [str(t) for t in self.doc if t not in self.get_punctuations()]

    def get_sentences(self):
        return [[t.text for t in sent] for sent in self.doc.sents]
    
    def normalize_text(self):
        return [
            str(t) for t in self.doc if 
                t not in self.get_stop_words()
                and t not in self.get_punctuations() 
                and str(t) in self.get_lemmas() 
        ]


text = "This Ruby Tuesday is in a strip mall at the corner of Routes 3 and 372. Parking seemed adequate when we were there but I'm not sure how it is when the restaurant is busy. What am I saying? Ruby Tuesday is never busy.This is a typical Ruby Tuesday where the all-you-can-eat salad bar is the prime attraction and is far more interesting than any of the entrees. If you go to Ruby Tuesday, I recommend ordering one of the mini sandwich combos with the salad bar because that's only a dollar more than ordering the salad bar as an entree. The restaurant interior decor is low key, comfortable and boring, the standard for this restaurant chain. Restrooms were kept in good shape, as far as I could tell. Service was pretty good."

text = TextPreprocessor(text, nlp)

# %%


len()

# %%
review_df = pd.read_csv("rt_reviews.csv")

# %%
df = review_df[["review_id", "review_text"]]

# %%

df["norm_review_text"] = df["review_text"].apply(lambda x: " ".join(TextPreprocessor(x, nlp).normalize_text()))
df["word_tokens"] = df["review_text"].apply(lambda x: TextPreprocessor(x, nlp).get_tokens())
df["review_word_count"] = df["word_tokens"].apply(lambda x: len(x))

# %%
sns.barplot(y="review_rating", data=review_df)
sns.histplot(data=df, x="review_word_count")
# %%

review_df.loc[1, "review_text"]