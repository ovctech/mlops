import os

import pandas as pd
from model import LitResnet
from prepare_dataset import prepare_dataset_test
from pytorch_lightning import Trainer


def infer():
    model = LitResnet.load_from_checkpoint(
        checkpoint_path="models/last_checkpoint.ckpt"
    )
    test_dataloader = prepare_dataset_test()
    trainer = Trainer()
    trainer.test(model, test_dataloader)
    predictions_df = pd.DataFrame({"prediction": model.test_predictions})
    os.makedirs("results", exist_ok=True)
    predictions_df.to_csv("results/test_predictions.csv", index=False)


if __name__ == "__main__":
    infer()
