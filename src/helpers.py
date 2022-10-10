import re
import numpy as np
import zipcodes


def get_zip_from_address(address: str) -> str:
    """
    Searches for US zip code format in full address
    """
    us_zip = r"(\d{5}\-?\d{0,4})"
    try:
        return re.search(us_zip, address).group(1)
    except AttributeError:
        return np.nan


def verify_zip_code(zip_code: str) -> bool:
    """
    Uses the zipcodes package to verify if a zip code is valid
    """
    return zipcodes.is_real(zip_code)


def generate_query(customer_code: str, address: str) -> dict:
    """
    Compiles the query string to get the business information given a
    customer_code and address sourced from the {env}_{customer_code}.stores
    standardized table.
    """
    try:
        zip_code = get_zip_from_address(address)
    except AttributeError:
        raise ValueError(f"No zip code found in the address '{address}'")

    return yelp_store_query(customer_code, zip_code)
