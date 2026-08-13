from core.datasets.hfdataloader import CSVHFDataset


class VMFunc4ChanPolExtensive(CSVHFDataset):
    def __init__(self, force_online=False):
        super().__init__("vmfunc/4chan-pol-extensive", csv_file=None, force_online=force_online)
