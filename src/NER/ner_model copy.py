# %%
import pandas as pd
import brightloompy.s3 as bpy
from random import sample, shuffle
import re
from typing import Tuple
import os
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

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

def generate_entities(entity_template, entitiy_words):
    entities = []
    for _ in range(0, 2000):
        template_sentence = sample(entity_template, 1)[0]

        num_entities_to_fill = len(re.findall("{}", template_sentence))
        
        entities_to_fill = sample(entitiy_words, num_entities_to_fill)

        filled_sentence = populate_template_sentence(template_sentence, entities_to_fill)

        entities.append(compile_entities(entity_type, filled_sentence, entities_to_fill))

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
        doc.ents = ents # label the text with the ents
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

# only keep 25% of the two worded and three worded foods; I want to make sure the single worded foods take priority
total_food_entities = round(len(food_one_words.list) / 50 * 100)
food_two_words.list = sample(food_two_words.list, round(total_food_entities * .25))
food_three_words.list = sample(food_three_words.list, round(total_food_entities * .25))

food_words = food_one_words.list + food_two_words.list + food_three_words.list
shuffle(food_words)

entities = generate_entities(food_entity_template, food_words)



TRAIN_DATA = sample(entities, 500)
TEST_DATA = [entity for entity in entities if entity not in TRAIN_DATA]

# %%
TRAIN_DATA.append(("I have a car and basketball with Shaq was a incredible", {"entities": []}))

# %%
nlp = spacy.load("en_core_web_lg")

create_spacy_data_file(nlp, TRAIN_DATA, save_path="./train.spacy")
create_spacy_data_file(nlp, TEST_DATA, save_path="./test.spacy")

#os.system("python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./test.spacy")


# %%
import spacy
nlp = spacy.load("en_core_web_lg")
food_nlp = spacy.load("./output/model-best")

food_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])

nlp.add_pipe('ner', source=food_nlp, name="food_nlp", after="ner")

print(nlp.pipe_names)

# %%
from spacy.tokens import Doc, Span
doc = nlp("Skip the turkey burger. Definitely came from a box.Unidentifiable as turkey or a burger. Service was ok.")
doc = nlp("The car and basketball with Shaq was a incredible") 
doc = nlp("Today with Shaq, I had a donut and apple inside a car in Brazil and it was great. The maple syrup was also good. So was the 10  piece chicken meal from Ruby Tuesday.")
doc = nlp("My husband and I recently dined at Ruby Tuesday, Bay Street, Taunton.")
spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter

# %%
for token in doc:
    print(token.text)

# %%
doc[0].vector
# %%
for token in doc:
    print(token, token.pos_,)

# %%
doc = nlp("Today I went to the Brazil and ate a pizza") # input sample text

spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter

# %%
TRAIN_DATA = sample(entities, 1500)
TEST_DATA = [entity for entity in entities if entity not in TRAIN_DATA]

TEST_DATA

# %%


# %%

TEST_FOOD_DATA = {
    "one_food": [entity for entity in entities if len(entity[1]["entities"]) == 1],
    "two_foods": [entity for entity in entities if len(entity[1]["entities"]) == 2],
    "three_foods": [entity for entity in entities if len(entity[1]["entities"]) == 3],
}
# %%
TEST_FOOD_DATA

# %%
food_evaluation = {
    "one_food": {
        "correct": 0,
        "total": 0,
    },
    "two_foods": {
        "correct": 0,
        "total": 0
    },
    "three_foods": {
        "correct": 0,
        "total": 0
    }
}

word_evaluation = {
    "1_worded_foods": {
        "correct": 0,
        "total": 0
    },
    "2_worded_foods": {
        "correct": 0,
        "total": 0
    },
    "3_worded_foods": {
        "correct": 0,
        "total": 0
    }
}

# loop over data from our test food set (3 keys in total)
for key in TEST_FOOD_DATA:
    foods = TEST_FOOD_DATA[key]

    for food in foods:
        # extract the sentence and correct food entities according to our test data
        sentence = food[0]
        entities = food[1]["entities"]

        # for each entity, use our updated model to make a prediction on the sentence
        for entity in entities:
            doc = nlp(sentence)
            correct_text = sentence[entity[0]:entity[1]]
            n_worded_food =  len(correct_text.split())

            # if we find that there's a match for predicted entity and predicted text, increment correct counters
            for ent in doc.ents:
                if ent.label_ == entity[2] and ent.text == correct_text:
                    food_evaluation[key]["correct"] += 1
                    if n_worded_food > 0:
                        word_evaluation[f"{n_worded_food}_worded_foods"]["correct"] += 1
                    
                    # this break is important, ensures that we're not double counting on a correct match
                    break
            
            #  increment total counters after each entity loop
            food_evaluation[key]["total"] += 1
            if n_worded_food > 0:
                word_evaluation[f"{n_worded_food}_worded_foods"]["total"] += 1


# %%
for key in word_evaluation:
    correct = word_evaluation[key]["correct"]
    total = word_evaluation[key]["total"]

    print(f"{key}: {correct / total * 100:.2f}%")

food_total_sum = 0
food_correct_sum = 0

print("---")
for key in food_evaluation:
    correct = food_evaluation[key]["correct"]
    total = food_evaluation[key]["total"]
    
    food_total_sum += total
    food_correct_sum += correct

    print(f"{key}: {correct / total * 100:.2f}%")

print(f"\nTotal: {food_correct_sum/food_total_sum * 100:.2f}%")


# %%
from spacy.lang.en import English
import spacy
nlp = English()
ruler = nlp.add_pipe("entity_ruler")
patterns = [{"label": "SERVICE", "pattern": [{"LOWER": "waiter"}]},]
ruler.add_patterns(patterns)


doc = nlp("The Waiter was pretty good")
print([(ent.text, ent.label_) for ent in doc.ents])
spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter
