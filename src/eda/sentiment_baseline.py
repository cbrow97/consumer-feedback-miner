# %%
import pandas as pd
from collections import Counter
import spacy
from collections import Counter
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from flair.models import TextClassifier
from flair.data import Sentence

text_df = pd.read_pickle("/home/ubuntu/consumer-feedback-miner/src/pre_process/cleaned_review_text.pkl")


nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("spacytextblob")

text_df["review_sentences"] = text_df["review_text"].apply(lambda x: [str(sent) for sent in nlp(x).sents])



# %%
def get_category_rating(polarity_score):
    if polarity_score < -0.25:
        return "negative"
    elif polarity_score > 0.25:
        return "positive"
    else:
        return "neutral"

def get_flair_category_rating(flair_score):
    if flair_score < -0.70:
        return "negative"
    elif flair_score > 0.70:
        return "positive"
    else:
        return "neutral"


# %%
text_df.drop(index=262, inplace=True)
# %%
classifier = TextClassifier.load('en-sentiment')
# %%
text = "i am neutral about cheese, it's not good good, it's not great"
text = Sentence(text)
classifier.predict(text)

value = text.labels[0].to_dict()['value'] 

# %%
text.to_dict()["all labels"][0]["confidence"]

# %%
def get_flair_review_sentence_sentiment(review_sentences):
    sentence_polarity = []
    for sent in review_sentences:
        current_sent = Sentence(sent)
        classifier.predict(current_sent)
        try:
            if current_sent.labels[0].to_dict()['value'] == "POSITIVE":
                sentence_polarity.append(current_sent.to_dict()["all labels"][0]["confidence"])
            else:
                sentence_polarity.append(-1 * current_sent.to_dict()["all labels"][0]["confidence"])
        except IndexError:
            pass
    avg_polarity = np.mean(sentence_polarity)
    return get_category_rating(avg_polarity)

text_df["flair_sentence_rating"] = text_df["review_sentences"].apply(lambda x: get_flair_review_sentence_sentiment(x))


# %%
def get_textblob_review_sentence_sentiment(review_sentences):
    sentence_polarity = []
    for sent in review_sentences:
        doc = nlp(sent)
        sentence_polarity.append(doc._.blob.sentiment.polarity)
    
    avg_polarity = np.mean(sentence_polarity)
    return get_category_rating(avg_polarity)

text_df["textblob_sentence_rating"] = text_df["review_sentences"].apply(lambda x: get_textblob_review_sentence_sentiment(x))

# %%
def get_vader_sentence_sentiment(review_sentences):
    sentence_polarity = []
    for sent in review_sentences:
        sid_obj = SentimentIntensityAnalyzer()
        sentence_polarity.append(sid_obj.polarity_scores(sent)["compound"])
    
    avg_polarity = np.mean(sentence_polarity)
    return get_category_rating(avg_polarity)

text_df["vader_sentence_rating"] = text_df["review_sentences"].apply(lambda x: get_vader_sentence_sentiment(x))

# %%
def get_flair_sentiment(review_text):
    review_text = Sentence(review_text)
    classifier.predict(review_text)

    if review_text.labels[0].to_dict()['value'] == "POSITIVE":
        return get_flair_category_rating(review_text.to_dict()["all labels"][0]["confidence"])
    else:
        return get_flair_category_rating(-1 * review_text.to_dict()["all labels"][0]["confidence"])


text_df["flair_rating"] = text_df["review_text"].apply(lambda x: get_flair_sentiment(x))

# %%
def get_textblob_sentiment(review_text):
    doc = nlp(review_text)
    return get_category_rating(doc._.blob.sentiment.polarity)

text_df["textblob_rating"] = text_df["review_text"].apply(lambda x: get_textblob_sentiment(x))

# %%
def get_vader_sentiment(review_text):
    sid_obj = SentimentIntensityAnalyzer()
    vader_rating = sid_obj.polarity_scores(review_text)["compound"]
    
    return get_category_rating(vader_rating)

text_df["vader_rating"] = text_df["review_text"].apply(lambda x: get_vader_sentiment(x))


# %%
labels = ["negative", "positive", "neutral"]

# %%
cf_flair_sentences = confusion_matrix(text_df["rating_category"], text_df["flair_sentence_rating"], labels=labels)

ax = make_confusion_matrix(cf_flair_sentences)
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# %%
cf_vader_sentences = confusion_matrix(text_df["rating_category"], text_df["vader_sentence_rating"], labels=labels)

ax = make_confusion_matrix(cf_vader_sentences)
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# %%
cf_vader = confusion_matrix(text_df["rating_category"], text_df["vader_rating"], labels=labels)

ax = make_confusion_matrix(cf_vader)
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# %%
cf_textblob = confusion_matrix(text_df["rating_category"], text_df["textblob_rating"], labels=labels)

ax = make_confusion_matrix(cf_textblob)
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# %%
cf_flair = confusion_matrix(text_df["rating_category"], text_df["flair_rating"], labels=labels)

ax = make_confusion_matrix(cf_flair)
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# %%
from sklearn.metrics import classification_report

print(classification_report(text_df["rating_category"], text_df["vader_rating"]))
print(classification_report(text_df["rating_category"], text_df["textblob_rating"]))

# %%
text_df[text_df["rating_category"] == "neutral"]
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

    return ax

