from typing import Union
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)


def get_dataset(
    dataset_name: str, split: str = "train"
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load a dataset from the Hugging Face datasets library.
    Args:
        dataset_name (`str`): The name of the dataset to load.
        split (`str`): The split of the dataset to load. Must be one of "train" or "validation".
    """
    assert split in ("train", "validation")
    return load_dataset(dataset_name, split=split)
