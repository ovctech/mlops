import subprocess

import config as cfg
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


def split_dataset(dataset, val_split=0.2, train=True):
    """Splits the dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(
        dataset, splits, generator=torch.Generator().manual_seed(42)
    )

    if train:
        return dataset_train
    return dataset_val


def get_splits(len_dataset, val_split):
    """Computes split lengths for train and validation set."""
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    else:
        raise ValueError(f"Unsupported type {type(val_split)}")

    return splits


def prepare_dataset_train():
    subprocess.run(["dvc", "pull"])

    cifar10_normalization = torchvision.transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    dataset_train = CIFAR10(
        root=cfg.PATH_DATASETS, train=True, download=False, transform=train_transforms
    )
    dataset_val = CIFAR10(
        root=cfg.PATH_DATASETS, train=True, download=False, transform=test_transforms
    )
    dataset_train = split_dataset(dataset_train)
    dataset_val = split_dataset(dataset_val, train=False)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_dataloader, val_dataloader


def prepare_dataset_test():
    subprocess.run(["dvc", "pull"])

    cifar10_normalization = torchvision.transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization,
        ]
    )
    dataset_test = CIFAR10(
        root=cfg.PATH_DATASETS, train=False, download=False, transform=test_transforms
    )

    test_dataloader = DataLoader(
        dataset_test,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return test_dataloader
