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
