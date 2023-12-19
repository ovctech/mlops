from pytorch_lightning import Trainer

from model import LitResnet
from prepare_data import train_dataloader, val_dataloader

model = LitResnet(lr=0.05)

trainer = Trainer(max_epochs=1, accelerator="auto", devices=1)

trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
