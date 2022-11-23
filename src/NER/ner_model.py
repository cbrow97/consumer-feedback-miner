"""
We need to get test sentences from real help data the include: 
    - food
    - service
    - location
    - amenities
"""
# %%
food_one_entity = [
    "Good {}",
    "all-you-can-eat {} bar is the prime attraction and is far more interesting than any of the entrees",
    "{} is a very good option",
    "So our {} was not good from start to finish",
    "The {} is a throwback feel",
    "{} are back",
    "That did not say delicious {} ahead",
    "The {} was not great",
    "The {} was great",
    "The {} was a massive disappointment",
    "The {} was a incredible",
    "I was being robbed that I had to pay for {} that was not as good as an average person would expect quality wise",
    "The place needs something more than just {} to win as a restaurant",
    "If I order {}, I specifically order them well done",
    "Yesterday I ordered the {} and I loved it",
    "I will say the {} is definitely not worth $15 though",
    "Good quality {}",
    "{} overall was good, as well as {}",
    "My friend ordered a {}",
    "My meal was a {}",
    "The {} alone drew us back again and again for a fresh and healthy meal",
    "Tonight we stopped in for a couple of {}", 
]

food_two_entities = [
    "I recommend ordering one of the {} with the {}",
    "The {} and {} was a incredible",
    "{} and {} is a very good option",
    "I really enjoyed the {} and {}",
    "Then I ordered {} and {}",
    "I suggest ordering a {} and adding the {} to it",
    "{} was less than good and the {} we ordered came out warmish",
    "Ordered {} with no {}",
    "My daughters friend ordered {} with {}",
    "The {} was tasteless and the {} was decent",
]

food_three_entities = [
    "brought out a plate of rock-hard {} looking things with dried out, stringy {} inside and a pile of underdone, whitish, {}",
    "The {} and {} with {} was a incredible",
    "We had the {} and {} with {}",
    "I had the {} and {} with {}",
]

service_one_entity = [
    "Good {}",
    "Fast {}",
    "{} seemed adequate when we were there",
    "So our {} was not good from start to finish",
    "The {} needs some real training",
    "{} looked at us but did nothing",
    "The {} was not great",
    "The {} however was a massive disappointment",
    "I was being robbed that I had to pay for {} that was not as good as an average person would expect quality wise",
    "If the kitchen is not producing {} of a high enough quality as prices are out of control people will not step up and open our wallets for mediocrity",
    "The {} was polite enough",
    "The {} intervened and asked what I wanted",
    "Our {} was super nice too",
    "The {} got my drink order wrong",
    "Good quality {}",
    "The {} was very friendly but seemed to be working too many tables",
    "The {} was extremely slow",
]

amenities_one_entity = [
    "{} seemed adequate when we were there",
    "More {} would be a plus",
    "The place needs something more than just {} to win as a restaurant",
]

amenities_two_entities = [
    "I really enjoyed the {} and {}",
    
]

amentities = [
    "seating",
    "parking",
    "comfort",
    "outdoor",
    "indoor",
    "wifi",
    "tv",
    "air conditioning",
    "reservations",
    "breakfast",
    "delivery",
    "bar",
]

food_entity_template = [
    "Good {}",
    "all-you-can-eat {} bar is the prime attraction and is far more interesting than any of the entrees",
    "{} is a very good option",
    "So our {} was not good from start to finish",
    "The {} is a throwback feel",
    "{} are back",
    "That did not say delicious {} ahead",
    "The {} was not great",
    "The {} was great",
    "The {} was a massive disappointment",
    "The {} was a incredible",
    "I was being robbed that I had to pay for {} that was not as good as an average person would expect quality wise",
    "The place needs something more than just {} to win as a restaurant",
    "If I order {}, I specifically order them well done",
    "Yesterday I ordered the {} and I loved it",
    "I will say the {} is definitely not worth $15 though",
    "Good quality {}",
    "{} overall was good, as well as {}",
    "My friend ordered a {}",
    "My meal was a {}",
    "The {} alone drew us back again and again for a fresh and healthy meal",
    "Tonight we stopped in for a couple of {}", 
    "brought out a plate of rock-hard {} looking things with dried out, stringy {} inside and a pile of underdone, whitish, {}",
    "The {} and {} with {} was a incredible",
    "We had the {} and {} with {}",
    "I had the {} and {} with {}",
    "I recommend ordering one of the {} with the {}",
    "The {} and {} was a incredible",
    "{} and {} is a very good option",
    "I really enjoyed the {} and {}",
    "Then I ordered {} and {}",
    "I suggest ordering a {} and adding the {} to it",
    "{} was less than good and the {} we ordered came out warmish",
    "Ordered {} with no {}",
    "My daughters friend ordered {} with {}",
    "The {} was tasteless and the {} was decent",
]

# %%
import pandas as pd

df = pd.read_csv("rt_reviews.csv")
' '.join(df["review_text"])

# %%

# %%
import pandas as pd
import brightloompy.s3 as bpy
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample, shuffle
import re
from typing import Tuple

food_df = bpy.read_csv("sandbox/colton/food.csv")

entity_type = "FOOD"

# %%
food_df["description"] = food_df["description"].fillna("")
food_df[food_df["description"].str.contains("little")]

# %%
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


# %%
food_one_words = PrepareEntities(food_df, "description", 1)
food_two_words = PrepareEntities(food_df, "description", 2)
food_three_words = PrepareEntities(food_df, "description", 3)

# %%

sns.barplot(
        x=["One Word", "Two Word", "Three Word"],
        y=[len(food_one_words.list), len(food_two_words.list), len(food_three_words.list)],
        palette='hls',
        
    )
plt.ylabel("Entities")
plt.title("Entities by Word Count")

# %%


total_food_entities = round(len(food_one_words.list) / 50 * 100)
food_two_words.list = sample(food_two_words.list, round(total_food_entities * .25))
food_three_words.list = sample(food_three_words.list, round(total_food_entities * .25))

sns.barplot(
        x=["One Word", "Two Word", "Three Word"],
        y=[len(food_one_words.list), len(food_two_words.list), len(food_three_words.list)],
        palette='hls',
        
    )
plt.ylabel("Entities")
plt.title("Entities by Word Count")


# %%

food_words = food_one_words.list + food_two_words.list + food_three_words.list
shuffle(food_words)

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

entities = []
for _ in range(0, 1000):
    template_sentence = sample(food_entity_template, 1)[0]

    num_entities_to_fill = len(re.findall("{}", template_sentence))
    
    entities_to_fill = sample(food_words, num_entities_to_fill)

    filled_sentence = populate_template_sentence(template_sentence, entities_to_fill)

    entities.append(compile_entities(entity_type, filled_sentence, entities_to_fill))

# %%
len([entity for entity in entities if len(entity[1]) == 1])

len([entity for entity in entities if len(entity[1]["entities"]) == 3])





# %%
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

#nlp = spacy.blank("en") # load a new spacy model
nlp = spacy.load("en_core_web_lg")
db = DocBin() # create a DocBin object

TRAIN_FOOD_DATA_COMBINED = sample(entities, 500)
TEST_FOOD_DATA = [entity for entity in entities if entity not in TRAIN_FOOD_DATA_COMBINED]



for text, annot in tqdm(TEST_FOOD_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./test.spacy") # save the docbin object
# %%
import spacy

nlp = spacy.load("en_core_web_sm")
food_nlp = spacy.load("./output/model-best")

food_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

nlp.add_pipe('ner', source=food_nlp, name="food_nlp", after="ner")

print(nlp.pipe_names)


# %%
doc = nlp("The donut and apple with carrot was a incredible") # input sample text
#doc = nlp("The car and basketball with Shaq was a incredible") 
spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter


# %%
doc = nlp("Today I went to the Brazil and ate a pizza") # input sample text

spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter

# %%
nlp1.vocab