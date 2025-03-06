import logging
import os
import pickle


def get_pre_encoded_mappings() -> dict:
    """
    Returns the predefined mappings for pre-encoded columns.
    """
    # Predefined mappings for already encoded categorical columns
    pre_encoded_mappings = {
        "country": {
            1: "Australia", 2: "Austria", 3: "Belgium", 4: "British Virgin Islands",
            5: "Cayman Islands", 6: "Christmas Island", 7: "Croatia", 8: "Cyprus",
            9: "Czech Republic", 10: "Denmark", 11: "Estonia", 12: "unidentified",
            13: "Faroe Islands", 14: "Finland", 15: "France", 16: "Germany",
            17: "Greece", 18: "Hungary", 19: "Iceland", 20: "India", 21: "Ireland",
            22: "Italy", 23: "Latvia", 24: "Lithuania", 25: "Luxembourg", 26: "Mexico",
            27: "Netherlands", 28: "Norway", 29: "Poland", 30: "Portugal", 31: "Romania",
            32: "Russia", 33: "San Marino", 34: "Slovakia", 35: "Slovenia", 36: "Spain",
            37: "Sweden", 38: "Switzerland", 39: "Ukraine", 40: "United Arab Emirates",
            41: "United Kingdom", 42: "USA", 43: "biz (*.biz)", 44: "com (*.com)",
            45: "int (*.int)", 46: "net (*.net)", 47: "org (*.org)"
        },
        "colour": {
            1: "beige", 2: "black", 3: "blue", 4: "brown", 5: "burgundy", 6: "gray",
            7: "green", 8: "navy blue", 9: "of many colors", 10: "olive", 11: "pink",
            12: "red", 13: "violet", 14: "white"
        },
        "location": {
            1: "top left", 2: "top middle", 3: "top right",
            4: "bottom left", 5: "bottom middle", 6: "bottom right"
        },
        "model_photography": {1: "en face", 2: "profile"},
        "price_2": {1: "yes", 2: "no"},
    }
    
    logging.info("Loaded predefined encoded mappings")
    return pre_encoded_mappings



def save_encoded_mappings(mapping_dict: dict, folder_path: str = "support") -> str:
    """
    Saves the encoded column mappings as a .pkl file inside the `support/` folder.

    Args:
        mapping_dict (dict): Dictionary containing mappings for categorical columns.
        folder_path (str): Directory where the mappings file should be stored.

    Returns:
        str: Path of the saved mappings file.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)  # Ensure `support/` directory exists
        file_path = os.path.join(folder_path, "encoded_mappings.pkl")

        with open(file_path, "wb") as file:
            pickle.dump(mapping_dict, file)

        logging.info(f"Encoded mappings successfully saved at: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error while saving mappings: {e}")
        logging.exception("Full Exception Traceback:")
        raise e