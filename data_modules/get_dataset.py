from .normal import NormalDataset

_dataset_dit = {
    'normal': NormalDataset
}


def get_dataset(dataset_type):
    return _dataset_dit[dataset_type]
