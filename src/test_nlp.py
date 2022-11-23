# %%
import spacy

nlp = spacy.load("en_core_web_lg")

doc = nlp("George Washington was the first president of the United States.")

for ent in doc.ents:
    print(ent.text, " -- ", ent.label_)

# %%
import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api

model = api.load("word2vec-google-news-300")


# %%
len(model)

# %%
model.vector_size


# %%
model.most_similar("active")

# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

data = pd.read_csv("amazon_cells_labelled.txt", sep="\t", header=None)

X = data.iloc[:, 0]
y = data.iloc[:, 1]

vectorizer = CountVectorizer(stop_words="english")
tfidf = TfidfTransformer()

X_vec = vectorizer.fit_transform(X)
X_vec.todense()

X_tfidf = tfidf.fit_transform(X_vec)
X_tfidf = X_tfidf.todense()

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=0)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# %%
confusion_matrix(y_test, y_pred)

