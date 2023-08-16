import lightning.pytorch as pl
from datasets import load_dataset
import os
import torchvision.transforms as transforms
import argparse
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

import lightning.pytorch as pl
import lightning.pytorch.loggers.tensorboard as tb
from lightning.pytorch.plugins.environments import SLURMEnvironment

import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import relu
from torch.optim import Adam, AdamW
from datasets import load_dataset

import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gilbreth/dchawra/cache/'
os.environ['HF_HOME'] = '/scratch/gilbreth/dchawra/hfhome/'

MODEL_NAME = "distilgpt2"

class TextGenDataModule(pl.LightningDataModule):
    def __init__(self, model_name="gpt2", dataset="tiny_shakespeare", batch_size=128):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: str):
        # load prepared dataset from disk with path
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.data = load_dataset(self.dataset, cache_dir="/scratch/gilbreth/dchawra/cache/")
        
        self.data.set_transform(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True, # truncates text to max_length
                max_length=128, # max length of text (ideally this would be the max length of the model)
                padding="max_length", # pads to max_length if the line is less than the max length
                return_tensors="np", # returns tensors in np format
            ))

        # it is already split into train, test, and validation sets : https://huggingface.co/datasets/tiny_shakespeare
        # so we can just assign them here
        
        self.train_data = self.data["train"]
        self.val_data = self.data["validation"]
        self.test_data = self.data["test"]

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)


class DollyModule(pl.LightningModule):
    def __init__(self, model_name="gpt2", learning_rate=1e-4):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir="/scratch/gilbreth/dchawra/cache/")

    def forward(self, batch):
        return self.model(**batch).loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
#        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

def main(args):
    torch.set_float32_matmul_precision('medium')

    print("Loading dataset...")
    datamodule = TextGenDataModule(MODEL_NAME, "tiny_shakespeare", 32)
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Loading model...")

    trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy="ddp",
                         plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                         max_epochs=100,
                         max_steps=1000)    
    model = DollyModule(MODEL_NAME, learning_rate=1e-4)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader)

    # save model in huggingface format
    trainer.save_checkpoint("/scratch/gilbreth/dchawra/text-generation-webui/models/")
    print("Model saved.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # TRAIN
    main(args)




        
            

