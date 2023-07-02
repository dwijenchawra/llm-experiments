from torch.utils.data import DataLoader
import os
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import argparse

# # importing the dataset
# from datasets import load_dataset

# dataset = load_dataset("yelp_review_full")
# dataset["train"][100]

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

import lightning.pytorch as pl
import lightning.pytorch.loggers.tensorboard as tb
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal

import torch.nn as nn
import torch.nn.functional as F
from torch import relu
from torch.optim import Adam



class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)



def main(args):
    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()), batch_size=128)
    trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=2, strategy="ddp",
                         plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)])
    model = LitModel()

    logger = tb.TensorBoardLogger("lightning_logs", name="mnisttest")

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # TRAIN
    main(args)