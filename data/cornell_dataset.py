from data.base_dataset import BaseDataset

class CornellDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self,opt)