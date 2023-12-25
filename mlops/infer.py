import os

import pandas as pd
from model import LitResnet
from prepare_dataset import prepare_dataset_test
from pytorch_lightning import Trainer


def get_path_of_model_params():
    # returns path of latest trained model
    base_directory = "outputs"
    timestamped_dirs = os.listdir(base_directory)
    sorted_timestamped_dirs_day = sorted(
        timestamped_dirs,
        key=lambda x: os.path.getctime(os.path.join(base_directory, x)),
        reverse=True,
    )
    sorted_timestamped_dirs_time = sorted(
        os.listdir(base_directory + "/" + sorted_timestamped_dirs_day[0]),
        key=lambda x: os.path.getctime(
            os.path.join(base_directory + "/" + sorted_timestamped_dirs_day[0], x)
        ),
        reverse=True,
    )
    return f"{base_directory+'/'+sorted_timestamped_dirs_day[0]+'/'+sorted_timestamped_dirs_time[0]}/last_checkpoint.ckpt"


def infer():
    model = LitResnet.load_from_checkpoint(
        checkpoint_path="models/last_checkpoint.ckpt"
    )
    test_dataloader = prepare_dataset_test()
    trainer = Trainer()
    trainer.test(model, test_dataloader)
    predictions_df = pd.DataFrame({"prediction": model.test_predictions})
    predictions_df.to_csv("results/test_predictions.csv", index=False)


if __name__ == "__main__":
    infer()
