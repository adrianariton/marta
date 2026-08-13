from core.datasets.hfdataloader import CSVHFDataset
import pandas as pd
import numpy as np


class HarmBench(CSVHFDataset):
    def __init__(self, force_online=False):
        super().__init__(
            "HarmBench/harmbench", csv_file=None, force_online=force_online, split="test"
        )

    def get_first(self, n):
        df = self.df.copy()
        df["id"] = [f"hb_{i}" for i in range(1, len(df) + 1)]
        return df.head(n)
