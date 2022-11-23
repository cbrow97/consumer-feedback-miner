import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
from collections import defaultdict


def get_category_rating(polarity_score):
    if polarity_score < -0.25:
        return "negative"
    elif polarity_score > 0.25:
        return "positive"
    else:
        return "neutral"


class VaderSentiment:
    def __init__(self):
        self.sid_obj = SentimentIntensityAnalyzer()

    def predict_sentences(self, sentences):
        sentence_polarity = []
        for sentence in sentences:
            sentence_polarity.append(self.sid_obj.polarity_scores(sentence)["compound"])
        
        avg_polarity = np.mean(sentence_polarity)
        return get_category_rating(avg_polarity)


    def predict(self, text):
        vader_rating = self.sid_obj.polarity_scores(text)["compound"]
        
        return get_category_rating(vader_rating)


class TextBlobSentiment:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.add_pipe("spacytextblob")

    def predict_sentences(self, sentences: list):
        sentence_polarity = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            sentence_polarity.append(doc._.blob.sentiment.polarity)
        
        avg_polarity = np.mean(sentence_polarity)
        return get_category_rating(avg_polarity)


    def predict(self, text: str):
        doc = self.nlp(text)
        return get_category_rating(doc._.blob.sentiment.polarity)


class RobertaBaseSentiment:
    def __init__(self):
        self.labels = ['negative', 'neutral', 'positive']
        self.task = "sentiment"
        self.MODEL = f"cardiffnlp/twitter-roberta-base-{self.task}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)


    def encode_text(self, text: str):
        return self.tokenizer(text, truncation=True, max_length=511, return_tensors='pt')

    def evaluate_prediction(self, output):
        scores = softmax(output[0][0].detach().numpy())
        ranking = np.argsort(scores)[::-1]

        return scores, ranking

    def predict(self, text: str):
        encoded_input = self.encode_text(text)
        output = self.model(**encoded_input)
        _, ranking = self.evaluate_prediction(output)

        return self.labels[ranking[0]]

    def predict_sentences(self, sentences: list):
        sentiments = defaultdict(list)

        for sentence in sentences:
            encoded_input = self.encode_text(sentence)
            output = self.model(**encoded_input)
            scores, ranking = self.evaluate_prediction(output)

            sentiments[self.labels[ranking[0]]].append(scores[ranking[0]])
        
        sentiments = {label:sum(sentiment_score) for label, sentiment_score in sentiments.items()}
        return max(sentiments, key=sentiments.get)
