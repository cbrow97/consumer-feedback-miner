# %%
import requests
import urllib.parse
import businesses
import pandas as pd
from typing import List

class ReviewGetter:
    def __init__(self, yelp_business: businesses.YelpBusinessInfo):
        self.api_key = "d3b3aeda74cc14d9bca60f2a538dd978a6858ed4"
        self.url = "https://data.unwrangle.com/api/getter/?url={encoded_url}&page={page_number}&api_key={api_key}"
        self.yelp_business = yelp_business

    def encode_url(self, business_url: str) -> str:
        return urllib.parse.quote(business_url)

    def build_url(self, business_url: str, page_number: str) -> str:
        return self.url.format(
            encoded_url=self.encode_url(business_url),
            page_number=page_number,
            api_key=self.api_key,
        )

    def get_review_response(self, business_url: str, page_number: str) -> dict:
        return requests.request("GET", self.build_url(business_url, page_number))

    def get_reviews(self) -> List:
        responses = [self.get_review_response(self.yelp_business.business_url, 1)]
        no_of_pages = responses[0].json()["no_of_pages"]

        print(f"There are {no_of_pages} pages of reviews to iterate over for {self.yelp_business.business_url}")
        for page_number in range(no_of_pages, no_of_pages + 1):
            responses.append(self.get_review_response(self.yelp_business.business_url, page_number))        

        return responses

    def get_review_df(self) -> pd.DataFrame:
        review_column_rename_mapper = {
            "id": "review_id",
            "date": "review_date",
            "rating": "review_rating",
            "meta_data.author_contributions": "author_contributions",
            "meta_data.feedback.useful": "review_useful_count",
            "meta_data.feedback.funny": "review_funny_count",
            "meta_data.feedback.cool": "review_cool_count",
            "response.text": "business_response_text",
            "response.name": "business_response_name",
            "response.role": "business_response_role",
            "response.date": "review_response_date",
        }

        review_df = pd.DataFrame()
        responses = self.get_reviews()

        temp_df = pd.concat([pd.json_normalize(response.json()["reviews"]) for response in responses])
        temp_df["business_id"] = self.yelp_business.business_id
        review_df = pd.concat([review_df, temp_df]).reset_index(drop=True)
        review_df = review_df.rename(columns=review_column_rename_mapper)

        return review_df


business_name = "Ruby Tuesday"
addresses = [
    "1975 S. Highway 27   Somerset, KY   42501",
    "1441 Tamiami Trail, Space #995   Port Charlotte, FL   33948",
]



yelp_business_info = [
        businesses.get_yelp_business(business_name, address) for address in addresses
    ]

review_getter = ReviewGetter(yelp_business_info[0])
