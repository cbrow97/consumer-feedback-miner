# %%
import spacy

nlp = spacy.load("en_core_web_lg")

doc = nlp("George Washington was the first president of the United States.")

for ent in doc.ents:
    print(ent.text, " -- ", ent.label_)
