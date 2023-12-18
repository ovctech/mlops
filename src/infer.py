import seaborn as sn
from IPython.display import display

from prepare_data import test_dataloader, train_dataloader, val_dataloader
from trainer import model, trainer

if __name__ == "__main__":
    trainer.test(model, test_dataloader)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")