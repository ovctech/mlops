import subprocess

import hydra
import mlflow
from model import LitResnet
from omegaconf import DictConfig
from prepare_dataset import prepare_dataset_train
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def get_git_commit_id():
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_id = ""
    return commit_id


@hydra.main(config_path="../configs", config_name="config_cpu_1")
def train(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("http://127.0.1.1:8080")
    # Start MLflow run
    with mlflow.start_run():
        # Prepare dataset
        train_dataloader, val_dataloader = prepare_dataset_train()

        # Create model
        model = LitResnet(lr=cfg.model.lr)

        # Callback to save model
        checkpoint_callback = ModelCheckpoint(
            dirpath="../../../models/", filename="last_checkpoint"
        )

        # Create Trainer
        trainer = Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            callbacks=[checkpoint_callback],
        )

        # Commit_id
        commit_id = get_git_commit_id()
        mlflow.set_tag("git_commit_id", commit_id)

        # Params
        mlflow.log_params(
            {
                "model_params": cfg.model,
                "max_epochs": cfg.trainer.max_epochs,
                "accelerator": cfg.trainer.accelerator,
                "devices": cfg.trainer.devices,
            }
        )

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    train()
