SERVICE_entities = [
    "waiter",
    "waitress",
    "wait staff",
    "waiters",
    "table",
    "host",
    "hostess",
    "chef",
    "cook",
    "bus",
    "busser",
    "clean",
    "cleaning",
    "manager",
    "staff",
    "service",
    "check",
    "experience",
    "order",
    "ordered",
    "waiting",
    "bill",
    "establishment",
    "owner",
    "bathroom",
    "restroom",
    "serve",
    "party",
    "accommodate",
    "person",
    "people",
    "cleanliness",
    "bartender",
    "gentleman",
    "wait staff",
]

def add_service_ent(nlp, entities=SERVICE_entities):
    config = {"overwrite_ents": True}
    ruler = nlp.add_pipe("entity_ruler", config=config)
    patterns = [{"label": "SERVICE", "pattern": [{"LOWER": entity}]} for entity in entities]
    ruler.add_patterns(patterns)
    return nlp
