from core.datasets.hfdataloader import CSVHFDataset


class LAODoNotAnswer(CSVHFDataset):
    def __init__(self, force_online=False):
        super().__init__(
            "LibrAI/do-not-answer", csv_file=None, force_online=force_online, split="train"
        )
