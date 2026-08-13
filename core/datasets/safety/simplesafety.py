from core.datasets.hfdataloader import CSVHFDataset


class BertievidgenSimpleSafetyTests(CSVHFDataset):
    def __init__(self, force_online=False):
        super().__init__(
            "Bertievidgen/SimpleSafetyTests", csv_file=None, force_online=force_online, split="test"
        )
