from core.datasets.hfdataloader import JSONHFDataset


class SicariusUBWTapestries(JSONHFDataset):
    def __init__(self, force_online=False):
        super().__init__(
            "SicariusSicariiStuff/UBW_Tapestries", json_file=None, force_online=force_online
        )
