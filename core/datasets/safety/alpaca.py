from core.datasets.hfdataloader import CSVHFDataset


class AlpacaBench(CSVHFDataset):
    """
    Benign instruction-following dataset, used as a non-harmful control/baseline
    in place of adversarial benchmarks like OrBench.

    @misc{alpaca,
        title={Stanford Alpaca: An Instruction-following LLaMA model},
        author={Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois
                and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
        year={2023},
        publisher={GitHub},
        journal={GitHub repository},
    }
    """

    def __init__(self, force_online=False):
        super().__init__(
            "tatsu-lab/alpaca",
            csv_file=None,
            force_online=force_online,
            split="train",
            name=None,
        )

    def _with_prompt_column(self, df):
        # Alpaca rows are {instruction, input, output}. Build a single "prompt"
        # field so downstream code (which expects row["prompt"]) works unchanged.
        def make_prompt(row):
            if row.get("input"):
                return f"{row['instruction']}\n\n{row['input']}"
            return row["instruction"]

        df["prompt"] = df.apply(make_prompt, axis=1)
        return df

    def get_first(self, n=100):
        df = super().all_df()
        df = self._with_prompt_column(df)
        df["iid"] = df.index
        df["id"] = df["iid"].apply(lambda x: "alpaca_" + str(x))
        return df.head(n)

    def sample_randomly(self, n, random_state=1):
        df = super().all_df()
        df = self._with_prompt_column(df)
        df["iid"] = df.index
        df["id"] = df["iid"].apply(lambda x: "alpaca_" + str(x))
        return df.sample(n, random_state=random_state)