from core.datasets.hfdataloader import CSVHFDataset


class AdvBench(CSVHFDataset):
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
            "walledai/AdvBench", csv_file=None, force_online=force_online, split="train"
        )
