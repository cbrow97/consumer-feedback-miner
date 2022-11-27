# %%
import pandas as pd
from collections import Counter
import spacy
from collections import Counter

text_df = pd.read_pickle("/home/ubuntu/consumer-feedback-miner/src/pre_process/cleaned_review_text.pkl")

nlp = spacy.load("en_core_web_trf")

doc = nlp(" ".join(text_df["norm_text"]))

entity_counter = Counter([word.label_ for word in doc.ents])

entity_baseline_df = pd.DataFrame(
    data=list(
        zip(list(entity_counter.keys()), list(entity_counter.values()))
    ), columns=["entity", "frequency"]
).sort_values("frequency", ascending=False)

entity_baseline_df
