import json


def load_config(config_path: str) -> dict:
    """Load the configuration file specified by the user.

    In case of error, the function will raise an exception.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The configuration file as a dictionary.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return config
