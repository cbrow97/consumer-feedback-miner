import requests
import pandas as pd
import helpers
from difflib import SequenceMatcher
from typing import List
from helpers import get_zip_from_address

_URL = "https://api.yelp.com/v3/businesses/search"


class YelpBusinessInfo:
    """
    A class representation of the relevant elements returned from the Yelp Businesses
    Search API object. Coverts the entry into a pandas dataframe row using the
    get_yelp_business_df_rows property method.
    """

    def __init__(self, yelp_business):
        self.business_id = yelp_business["id"]
        self.alias = yelp_business["alias"]
        self.name = yelp_business["name"]
        self.business_url = self.business_url(yelp_business)
        self.address = self.address(yelp_business)
        self.rating = yelp_business["rating"]

    def business_url(self, yelp_business):
        return yelp_business["url"].split("?")[0]

    def address(self, yelp_business):
        return " ".join(yelp_business["location"]["display_address"])

    @property
    def get_yelp_business_df_row(self):
        return pd.DataFrame(self.__dict__, index=[0])


def yelp_header():
    key = "MpZk3PBR3w3ou5VHdgZXBUPfs_1fPg2suhJ1-2WwGsIt7R4LrV5gqvgY7VjUT9cpZmBVjqwhkqBD8nvMisAZVUUg6YaIxNHjvZI7u3gLIY77eQ1HEs-aBfLplLsHY3Yx"  # noqa
    return {"Authorization": f"Bearer {key}"}


def yelp_business_query(business_name, zip_code):
    return {
        "term": business_name,
        "location": zip_code,
        "radious": 10,
    }


def generate_business_query(business_name: str, address: str) -> dict:
    """
    Compiles the query string to get the business information given a
    business_name and address sourced from the {env}_{business_name}.stores
    standardized table.
    """
    try:
        zip_code = get_zip_from_address(address)
    except AttributeError:
        raise ValueError(f"No zip code found in the address '{address}'")

    return yelp_business_query(business_name, zip_code)


def get_business_response(
    response: dict, business_name: str, match_threshold: float
) -> List:
    """
    Matches the name of the business_name to the name and alias from
    the yelp API query results. Results are accepted based on the match_threshold
    when comparing business_name to name and alias
    """
    return [
        business
        for business in response.json()["businesses"]
        if SequenceMatcher(None, business_name, business["name"]).ratio()
        > match_threshold  # noqa
        or SequenceMatcher(None, business_name, business["alias"]).ratio()
        > match_threshold  # noqa
    ]


def get_yelp_business(business_name: str, address: str):
    response = requests.get(
        _URL,
        headers=yelp_header(),
        params=generate_business_query(business_name, address),
    )

    if response.status_code == 200:
        yelp_business = get_business_response(response, business_name, 0.8)

        if yelp_business is not None:
            return YelpBusinessInfo(yelp_business[0])
    else:
        print(
            f"No business found for business name: {business_name} at address: {address}."
        )


def generate_yelp_businesses_df(business_name: str, addresses: list):
    business_column_rename_mapper = {
        "rating": "business_rating",
        "alias": "business_alias",
        "name": "business_name",
        "address": "business_address",
    }

    yelp_business_info = [
        get_yelp_business(business_name, address) for address in addresses
    ]
    yelp_businesses_df = pd.concat(
        [yelp_business.get_yelp_business_df_row for yelp_business in yelp_business_info]
    ).reset_index(drop=True)
    yelp_businesses_df["business_zip_code"] = yelp_businesses_df.apply(
        lambda x: helpers.get_zip_from_address(x["address"]), axis=1
    )
    yelp_businesses_df = yelp_businesses_df.rename(
        columns=business_column_rename_mapper
    )
    return yelp_businesses_df


business_name = "Ruby Tuesday"
addresses = [
    "1975 S. Highway 27   Somerset, KY   42501",
    "1441 Tamiami Trail, Space #995   Port Charlotte, FL   33948",
]

generate_yelp_businesses_df(business_name, addresses)
