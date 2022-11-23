
# %%
#EDA Figure 1: Count of reviews overtime AND Count of reviews overtime by rating category

sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,10))
fig.tight_layout(pad=5)



fig_1_data = text_df.groupby("review_month_year")["review_id"].count().reset_index().rename(columns={"review_id": "review_count"})

sns.lineplot(x="review_month_year", y="review_count", data=fig_1_data, ax=ax1, c="purple")
ax1.set_title("Review Count by Month", fontsize=16)
ax1.set_xlabel("Month of Review", fontsize=16)
ax1.set_ylabel("Review Count", fontsize=16)
ax1.tick_params(axis="y", labelsize="16")

fig_2_data = text_df.groupby(["review_month_year", "rating_category"])["review_id"].count().reset_index().rename(columns={"review_id": "review_count"})

sns.lineplot(x="review_month_year", y="review_count", hue="rating_category", data=fig_2_data, style="rating_category", ax=ax2)
ax2.set_title("Review Count by Month and Rating Category", fontsize=16)
ax2.set_xlabel("Month of Review", fontsize=16)
ax2.set_ylabel("Review Count", fontsize=16)
ax2.legend(title="Rating Category")
ax2.tick_params(axis="y", labelsize="16")

# %%
#EDA figure 2: rating category spread
plt.figure(figsize=(9, 6))
rating_category_df = pd.DataFrame(Counter(text_df["rating_category"]).items(), columns=["rating_category", "count"])
total = rating_category_df["count"].sum()
ax = sns.barplot(x="rating_category", y="count", data=rating_category_df)

for p in ax.patches:
    percentage = "{:.1f}%".format(100 * p.get_height()/total)
    x = p.get_x() + (p.get_width() / 2)
    y = (p.get_height() / 2)
    ax.annotate(percentage + "\n of total", (x, y), ha="center")

ax.set(
    ylabel="Count",
    xlabel="Rating Category",
    title="Count of Rating Categories"
)
# %%
#EDA figure 3: word and sentence count
sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 8))
fig.tight_layout(pad=5)

sns.histplot(x=[text.word_count for text in processed_text.values()], ax=ax1)
ax1.set_title("Number of Words per Review", fontsize=16)
ax1.set_xlabel("Word Count", fontsize=16)
ax1.set_ylabel("Review Count", fontsize=16)
ax1.tick_params(axis="y", labelsize="16")

sns.histplot(x=[text.sentence_count for text in processed_text.values()], ax=ax2)
ax2.set_title("Number of Sentences per Review", fontsize=16)
ax2.set_xlabel("Sentence Count", fontsize=16)
ax2.set_ylabel("Review Count", fontsize=16)
ax2.tick_params(axis="y", labelsize="16")


# %%
#EDA figure 4: word frequency:
sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
fig.tight_layout(pad=5)

word_count = Counter(' '.join(t.norm_text for t in processed_text.values()).split())
word_count_df = pd.DataFrame(word_count.items(),columns=["word", "count"])
word_count_df = word_count_df.sort_values("count", ascending=False).reset_index(drop=True)
sns.barplot(word_count_df.head(25)["count"], word_count_df.head(25)["word"], ax=ax)
ax.set_title("25 Most Frequent Words", fontsize=16)
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Word", fontsize=16)
ax.tick_params(axis="y", labelsize="16")


# %%
#EDA figure 5: word frequency by rating category:
def get_word_count_by_rating_category(text_df: pd.DataFrame, rating_category: str) -> list:
    return Counter(' '.join(t.norm_text for t in processed_text.values() if t.rating_category == rating_category).split())

word_count_negative = get_word_count_by_rating_category(text_df, "negative")
word_count_neutral = get_word_count_by_rating_category(text_df, "neutral")
word_count_positive = get_word_count_by_rating_category(text_df, "positive")

# %%
sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
fig.tight_layout(pad=5)

#word_count = Counter(' '.join(t.norm_text for t in processed_text.values()).split())
word_count_df = pd.DataFrame(word_count_positive.items(),columns=["word", "count"])
word_count_df = word_count_df.sort_values("count", ascending=False).reset_index(drop=True)
sns.barplot(word_count_df.head(25)["count"], word_count_df.head(25)["word"], ax=ax)
ax.set_title("25 Most Frequent Words", fontsize=16)
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Word", fontsize=16)
ax.tick_params(axis="y", labelsize="16")


# %%

# %%
text_df["review_month_year"][0]
# %%
fig_1_data
# %%

# %%
sns.set_theme(style="white", palette="pastel")
plt.rc('font', size=18)
plt.figure(figsize=(9, 6))
sns.histplot(x="word_count", data=text_df, hue="rating_category", kde=True)


# %%




# %%
sns.barplot(y=[text.rating for text in processed_text.values()])

# %%
[text.rating for text in processed_text.values()]

# %%
plt.figure(figsize=(6, 4))
word_count_by_rating_cat_bp = (
    sns.boxplot(x="word_count", y="rating_category", data=text_df)
    .set(
        title="Review Word Count by Rating Category",
        xlabel="Word Count",
        ylabel="Rating Category",
    )
)

# %%


