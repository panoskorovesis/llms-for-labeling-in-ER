from src.utils.utils import load_config
import subprocess
from tqdm import tqdm


class Downloader:
    def __init__(self, url_prefix):
        """The constructor of the Downloader class.

        It sets the url_prefix and logger attributes to be used by
        the class methods.

        Args:
            url_prefix (str): The prefix of the URL to be used in the
                downloader.
            logger (logging.Logger): The logger object to be used for

        Returns:
            None
        """
        self.url_prefix = url_prefix

        # The following dictionary maps the folders with the
        # expected file prefix
        self.folder_file_mapping = {
            "D1": ["rest1", "rest2", "gt"],
            "D2": ["buy", "abt", "gt"],
            "D3": ["amazon", "gp", "gt"],
            "D4": ["acm", "dblp", "gt"],
            "D10": ["dbpedia", "imdb", "gt"],
        }

        print("Downloader initialized")

    def folder_is_valid(self, folder: str):
        """Check if the folder is valid.

        A folder is considered valid if it is in the folder_file_mapping

        Args:
            folder (str): The folder to be checked.

        Returns:
            bool: True if the folder is valid, False otherwise.
        """
        return folder in self.folder_file_mapping.keys()

    def download(self, folder: str, file_name: str):
        """Download the file specified by the URL suffix.

        The download will be done using subprocess and wget.

        Args:
            url_suffix (str): The suffix of the URL to be used in the
                downloader.

        Returns:
            None
        """

        # Download the file
        result = subprocess.run(
            ["wget", f"{self.url_prefix}/{folder}/{file_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # In case of error raise an exception
        if result.returncode != 0:
            raise Exception(
                f"Error downloading file: {result.stderr}. Command failed with status: {result.stdout}"
            )

        # Now move the downloaded file to the datasets folder

        # First create the folder if it does not exist
        result = subprocess.run(
            [
                "mkdir",
                "-p",
                f"./datasets/{folder}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # In case of error raise an exception
        if result.returncode != 0:
            raise Exception(
                f"Error moving file: {result.stderr}. Command failed with status: {result.stdout}"
            )

        # Move the file to the datasets folder
        result = subprocess.run(
            [
                "mv",
                f"{file_name}",
                f"./datasets/{folder}/{file_name}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # In case of error raise an exception
        if result.returncode != 0:
            raise Exception(
                f"Error moving file: {result.stderr}. Command failed with status: {result.stdout}"
            )

    def download_data(self, folders: list):
        """Download the data from the specified folders.

        We know that each folder contains some specific files.
        We are interested in the gt_clean and the clean domain files
        Each folder contains a specific domain such as
            restaurants, movies, products, etc.

        The mapping between the folder and the domain is not subject to change
        and is thus hardcoded in the folder_file_mapping dictionary.
        """

        for folder in tqdm(folders, desc="Downloading data", total=len(folders)):
            # If the requested folder is not in the mapping raise an exception
            if self.folder_is_valid(folder):
                # Download the files for the folder
                for file_name in self.folder_file_mapping[folder]:
                    self.download(folder, file_name + "clean.csv")
            else:
                raise ValueError(
                    f"Invalid folder: {folder}. Please provide any of {self.folder_file_mapping.keys()}"
                )


if __name__ == "__main__":
    config = load_config("src/config/config.json")

    # The required data are under https://github.com/AI-team-UoA/pyJedAI/tree/main/data/ccer
    # For our experiment we will need only the clean clean datasets
    # The following downloader will be used to download the required data
    # And save them in the datasets folder

    downloader = Downloader(
        url_prefix=config["data-downloader"]["url-prefix"],
    )

    downloader.download_data(config["data-downloader"]["required_datasets"])

    print("Data downloaded successfully")
