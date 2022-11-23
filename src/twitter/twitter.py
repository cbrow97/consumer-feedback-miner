# %%
import tweepy

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAAPVZiwEAAAAAPudjtCvbywWfxYXMe3rixTGUdVc%3DlTC9abbJsZ0SHUXWWYnJilXNgv4GURWfU4qIe5c8wgD8zoENK8"


client = tweepy.Client(bearer_token=bearer_token)

query = 'entity:"Ruby Tuesday" lang:en'
response = client.search_recent_tweets(query)

print(response)

# %%
# https://www.youtube.com/watch?v=0EekpQBEP_8&ab_channel=SuhemParack
# https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query