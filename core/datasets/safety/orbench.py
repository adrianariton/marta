# sorry-bench/sorry-bench-202503

from core.datasets.hfdataloader import CSVHFDataset


class OrBench(CSVHFDataset):
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
            "bench-llm/or-bench",
            csv_file=None,
            force_online=force_online,
            split="train",
            name="or-bench-hard-1k",
        )

    def get_first(self, n=100):
        df = super().all_df()
        df["iid"] = df.index
        df["id"] = df["iid"].apply(lambda x: "orbench_" + str(x))
        
        return df.head(n)
    
    def sample_randomly(self, n, random_state = 1):
        df = super().all_df()
        df["iid"] = df.index
        df["id"] = df["iid"].apply(lambda x: "orbench_" + str(x))
        return df.sample(n, random_state=random_state)
