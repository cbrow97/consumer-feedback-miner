# %%
import pandas as pd
import brightloompy.s3 as bpy
from random import sample, shuffle
import re
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import seaborn as sns
import matplotlib.pyplot as plt
from random_nouns import noun_list
from spacy.util import filter_spans
from food_entity_template_senteces import food_entity_template
import string

class PrepareEntities:
    def __init__(self, df:pd.DataFrame, field:str, words_in_entity:int):
        entity_series = self.normalize_entities(df, field)
        self.list = self.filter_n_words(entity_series, words_in_entity)

    def normalize_entities(self, df:pd.DataFrame, field:str) -> pd.DataFrame:
        """
        - removes values with any special characters
        - lowercase values
        """
        return df[
            df[field].str.contains("[^a-zA-Z ]") == False
        ][field].apply(lambda food: food.lower())

    def filter_n_words(self, entity_series:pd.Series, words_in_entity: str):
        return list(entity_series[entity_series.str.split().apply(len) == words_in_entity].drop_duplicates())


def populate_template_sentence(template_sentence: str, entities_to_fill: list) -> str:
    """
    Replaces instances of {} in the template_sentence with the values in
    the entities_to_fill list.

    E.g.
        Given the inputs:
            template_sentence = "I really enjoyed the {} and {}"
            entities_to_fill = ["chicken wings", "tacos"]

        The returned string would be:
            "I really enjoyed the chicken wings and tacos"
    """
    for entity in entities_to_fill:
        position = template_sentence.find("{}")
        template_sentence = template_sentence[:position] + entity + template_sentence[position+2:]
    
    return template_sentence


def compile_entities(entity_type: str, filled_sentence: str, entities_to_fill: list) -> Tuple[str, dict]:
    """
    Compiles the entities within a populated template_sentence in a format that is
    expected when training a spaCy NER model.

    E.g.
        Given the inputs:
            entity_type = "FOOD"
            filled_sentence = "I really enjoyed the chicken wings and tacos"
            entities_to_fill = ["chicken wings", "tacos"]

        The returned string would be:
            ('I really enjoyed the chicken wings and tacos',
            {'entities': [(21, 33, 'FOOD'), (39, 43, 'FOOD')]})      
    """
    return (
        filled_sentence,
        {
            "entities": [
                (
                    *re.search(entity, filled_sentence).span(),
                    entity_type
                ) for entity in entities_to_fill
                ]
        }
    )

def generate_food_entities(entity_template, entitiy_words):
    entities = []
    for _ in range(0, 2000):
        template_sentence = sample(entity_template, 1)[0]

        num_entities_to_fill = len(re.findall("{}", template_sentence))
        
        entities_to_fill = sample(entitiy_words, num_entities_to_fill)

        filled_sentence = populate_template_sentence(template_sentence, entities_to_fill)

        entities.append(compile_entities(entity_type, filled_sentence, entities_to_fill))

    return entities

def remove_punctuation_entities(doc):
    out = []
    for ent in doc.ents:
        if ent[-1].text in string.punctuation:
            out.append(ent[0:-1])
        else:
            out.append(ent)
    doc.ents = out
    return doc


def generate_blank_entities(entity_template, entitiy_words):
    entities = []
    for _ in range(0, 1000):
        template_sentence = sample(entity_template, 1)[0]
        
        num_entities_to_fill = len(re.findall("{}", template_sentence))
        
        entities_to_fill = sample(entitiy_words, num_entities_to_fill)

        filled_sentence = populate_template_sentence(template_sentence, entities_to_fill)

        entities.append(compile_entities("", filled_sentence, ""))

    return entities


def create_spacy_data_file(nlp, data, save_path=None):
    db = DocBin()

    for text, annot in tqdm(data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        #doc.ents = ents # label the text with the ents
        doc.ents = filter_spans(ents)
        db.add(doc)

    if save_path:
        db.to_disk(save_path)
    else:
        return db

food_df = bpy.read_csv("sandbox/colton/food.csv")
entity_type = "FOOD"

food_one_words = PrepareEntities(food_df, "description", 1)
food_two_words = PrepareEntities(food_df, "description", 2)
food_three_words = PrepareEntities(food_df, "description", 3)


sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.tight_layout(pad=3)

sns.barplot(
        x=["One Word", "Two Word", "Three Word"],
        y=[len(food_one_words.list), len(food_two_words.list), len(food_three_words.list)],
        ax=ax1
    )
ax1.set_ylabel("Number of Food Entries")
ax1.set_xlabel("Entry Word Count")
ax1.set_title("Entries by Word Count")

# only keep 25% of the two worded and three worded foods; I want to make sure the single worded foods take priority
total_food_entities = round(len(food_one_words.list) / 50 * 100)
food_two_words.list = sample(food_two_words.list, round(total_food_entities * .35))
food_three_words.list = sample(food_three_words.list, round(total_food_entities * .15))

sns.barplot(
        x=["One Word", "Two Word", "Three Word"],
        y=[len(food_one_words.list), len(food_two_words.list), len(food_three_words.list)],
        ax=ax2
    )
ax2.set_ylabel("Number of Food Entries")
ax2.set_xlabel("Entry Word Count")
ax2.set_title("Entries by Word Count After Fixing Skew")


food_words = food_one_words.list + food_two_words.list + food_three_words.list
shuffle(food_words)

food_entities = generate_food_entities(food_entity_template, food_words)
blank_entities = generate_blank_entities(food_entity_template, noun_list)

TRAIN_DATA = sample(food_entities, 1000) + sample(blank_entities, 500)
TRAIN_DATA.append(("Today I got into a car", {'entities': []}))

TEST_DATA = (
    [entity for entity in food_entities if entity not in TRAIN_DATA] + 
    [entity for entity in blank_entities if entity not in TRAIN_DATA]
)

shuffle(TRAIN_DATA)
shuffle(TEST_DATA)


nlp = spacy.load("en_core_web_trf")

create_spacy_data_file(nlp, TRAIN_DATA, save_path="./train.spacy")
create_spacy_data_file(nlp, TEST_DATA, save_path="./test.spacy")

#os.system("python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./test.spacy")


# %%
import spacy
from ner_SERVICE import add_service_ent

nlp = spacy.load("en_core_web_trf")
food_nlp = spacy.load("./output/model-best")

food_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

nlp.add_pipe('ner', source=food_nlp, name="food_nlp", after="ner")

print(nlp.pipe_names)

add_service_ent(nlp)

doc = nlp("Today with Shaq, I had a donut and apple inside a car in Brazil and it was great. The maple syrup was also good. So was the 10  piece chicken meal from Burger Palace.")
spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter
