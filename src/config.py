import os

import torch

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./../data/cifar-10-batches-py/")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
