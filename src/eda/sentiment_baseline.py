# %%
from sentiment.sentiment_models import VaderSentiment, TextBlobSentiment, RobertaBaseSentiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
                          title=None,
                          ax=None):
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
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,xticklabels=categories,yticklabels=categories, ax=ax, annot_kws={"size": 16})

    if xyplotlabels:
        ax.set_ylabel('True label', fontsize=16)
        ax.set_xlabel('Predicted label' + stats_text, fontsize=16)
    else:
        ax.set_xlabel(stats_text, fontsize=16)

    if title:
        ax.set_title(title, fontsize=16)

# %%
text_df = pd.read_pickle("/home/ubuntu/consumer-feedback-analyzer/src/pre_process/cleaned_review_text.pkl")
text_df = text_df[["review_text", "review_sentences", "rating_category"]]
index_to_drop = text_df[text_df["rating_category"] == "negative"].sample(n=492-232).index
text_df = text_df.drop(index=index_to_drop)

# %%
sentiment_score_mapper = {
    "negative": "negative",
    "somewhat negative": "negative",
    "neutral": "neutral",
    "somewhat positive": "positive",
    "positive": "positive",
}

labels = ['negative', 'positive', 'neutral']

labeled_df = pd.read_csv("labeled_review_sentences.csv")
labeled_df = labeled_df[~labeled_df["label"].isna()]
labeled_df["rating_category"] = labeled_df["label"].str.lower().map(sentiment_score_mapper)
labeled_df["review_sentences"] = labeled_df["review_sentences"].apply(lambda x: x[0:-1].split(','))

# %%
model = RobertaBaseSentiment()
model_type = "bert"
labeled_df[f"{model_type}_sentence_sentiment"] = labeled_df["review_sentences"].apply(lambda x: model.predict_sentences(x))

# %%
fig, ax1 = plt.subplots(1, 1, figsize=(17, 6))
fig.tight_layout(pad=5)
cf = confusion_matrix(labeled_df["rating_category"], labeled_df[f"{model_type}_sentence_sentiment"], labels=labels)
make_confusion_matrix(cf, title=f"{model_type.title()} Model Performance", cbar=False, categories=labels, ax=ax1)


# %%
models = {
    "vader": VaderSentiment(),
    "textblob": TextBlobSentiment(),
    "bert": RobertaBaseSentiment(),
}

for model_type, model in models.items():
    text_df[f"{model_type}_sentiment"] = text_df["review_text"].apply(lambda x: model.predict(x))
    text_df[f"{model_type}_sentence_sentiment"] = text_df["review_sentences"].apply(lambda x: model.predict_sentences(x))


sns.set(font_scale=1.2)
labels = ['negative', 'positive', 'neutral']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
fig.tight_layout(pad=5)

for model_type, ax in zip(models.keys(), (ax1, ax2, ax3)):
    cf = confusion_matrix(text_df["rating_category"], text_df[f"{model_type}_sentiment"], labels=labels)
    make_confusion_matrix(cf, title=f"{model_type.title()} Model Performance", cbar=False, categories=labels, ax=ax)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))
fig.tight_layout(pad=5)

for model_type, ax in zip(models.keys(), (ax1, ax2, ax3)):
    cf = confusion_matrix(text_df["rating_category"], text_df[f"{model_type}_sentence_sentiment"], labels=labels)
    make_confusion_matrix(cf, title=f"{model_type.title()} Model Performance", cbar=False, categories=labels, ax=ax)

