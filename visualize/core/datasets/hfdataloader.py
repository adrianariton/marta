import pandas as pd
import os
from datasets import load_dataset
import json


class CSVHFDataset:
    def __init__(
        self,
        ds_path,
        csv_file=None,
        force_online=False,
        split="train",
        **kwargs
    ):
        self.ds_path = ds_path
        self.csv_file = csv_file or f"local/{self.ds_path}.csv"
        self.split = split

        if force_online or not os.path.exists(self.csv_file):
            print(f"Downloading {self.ds_path} from Hugging Face...")
            ds = load_dataset(self.ds_path, split=split, **kwargs)
            self.df = ds.to_pandas()

            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            self.df.to_csv(self.csv_file, index=False)
        else:
            print(f"Loading {self.ds_path} from local cache: {self.csv_file}")
            self.df = pd.read_csv(self.csv_file)

    def sample_randomly(self, n: int, random_state: int = 1) -> pd.DataFrame:
        return self.df.sample(n=n, random_state=random_state)

    def get_first(self, n: int) -> pd.DataFrame:
        """Returns the first n rows of the dataframe."""
        return self.df.head(n)

    def all_df(self) -> pd.DataFrame:
        """Returns the entire dataframe."""
        return self.df


class JSONHFDataset:
    def __init__(self, ds_path, json_file=None, force_online=False, split="train"):
        self.ds_path = ds_path
        self.json_file = json_file or f"local/{self.ds_path}.json"
        self.split = split

        if force_online or not os.path.exists(self.json_file):
            print(f"Downloading {self.ds_path} from Hugging Face...")
            ds = load_dataset(self.ds_path, split=split)
            self.df = ds.to_pandas()

            os.makedirs(os.path.dirname(self.json_file), exist_ok=True)
            self.df.to_json(self.json_file, index=False)
        else:
            print(f"Loading {self.ds_path} from local cache: {self.json_file}")
            self.df = pd.read_json(self.json_file)

    def sample_randomly(self, n: int, random_state: int = 1) -> pd.DataFrame:
        return self.df.sample(n=n, random_state=random_state)

    def get_first(self, n: int) -> pd.DataFrame:
        """Returns the first n rows of the dataframe."""
        return self.df.head(n)

    def all_df(self) -> pd.DataFrame:
        """Returns the entire dataframe."""
        return self.df


# Usage
# dataset = HFDataset("vmfunc/4chan-pol-extensive")
