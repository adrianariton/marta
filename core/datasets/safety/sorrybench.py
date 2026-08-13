# sorry-bench/sorry-bench-202503

from core.datasets.hfdataloader import CSVHFDataset


class SorryBench(CSVHFDataset):
    """
    @misc{zou2023universal,
        title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
        author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
        year={2023},
        eprint={2307.15043},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """

    def __init__(self, force_online=False):
        super().__init__(
            "sorry-bench/sorry-bench-202503",
            csv_file=None,
            force_online=force_online,
            split="train",
        )

    def get_first(self, n=100):
        df = super().all_df()
        df["id"] = df["question_id"].apply(lambda x: "sorry_" + str(x))
        df["prompt"] = df["turns"].apply(
            lambda x: type(x) == str and eval(x)[0] if type(x) == str else ""
        )
        df_by_category = df.groupby("category").head(n // len(df["category"].unique()))
        joined_df = df_by_category.sort_values("category").reset_index(drop=True)
        return joined_df
