from core.datasets.hfdataloader import CSVHFDataset
import pandas as pd
import numpy as np


class XSTest(CSVHFDataset):
    def __init__(self, force_online=False):
        super().__init__("walledai/XSTest", csv_file=None, force_online=force_online, split="test")

    def get_first(self, n):
        df = self.df.copy()
        df_safe = df[df["label"] == "safe"].copy()
        df_safe.reset_index(drop=True, inplace=True)
        df_safe["id"] = "xstest_" + df_safe["label"] + "_" + df_safe.index.astype(str).str.zfill(3)
        df_unsafe = df[df["label"] == "unsafe"].copy()
        df_unsafe.reset_index(drop=True, inplace=True)
        df_unsafe["id"] = (
            "xstest_" + df_unsafe["label"] + "_" + df_unsafe.index.astype(str).str.zfill(3)
        )
        # df = pd.concat([df_safe, df_unsafe], axis=0)
        # df = df.reset_index(drop=True)

        df_safe["temp_idx"] = np.arange(len(df_safe)) // 10
        df_unsafe["temp_idx"] = np.arange(len(df_unsafe)) // 10

        # 2. Add a flag to keep the order within the chunks (safe first, then unsafe)
        df_safe["temp_order"] = 0
        df_unsafe["temp_order"] = 1

        # 3. Combine and sort
        df = pd.concat([df_safe, df_unsafe], axis=0)
        df = df.sort_values(["temp_idx", "temp_order"]).reset_index(drop=True)

        # 4. Clean up the helper columns
        df = df.drop(columns=["temp_idx", "temp_order"])

        return df.head(n)
