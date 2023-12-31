import importlib
import torch
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.")
    return dataset


def create_dataset(opt, mode):
    data_loader = CustomDatasetDataLoader(opt, mode)
    return data_loader


class CustomDatasetDataLoader():
    def __init__(self, opt, mode):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_name)
        self.dataset = dataset_class(opt, mode)
        print(f"dataset {type(self.dataset).__name__} {mode}was created!")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size if mode == "train" else 1,
            shuffle=opt.shuffle,
            num_workers=int(opt.num_threads)
        )

    def __iter__(self):
        for data, label, idx, rot, zoom_factor in self.dataloader:
            yield data, label, idx, rot, zoom_factor

    def __len__(self):
        return len(self.dataset)
