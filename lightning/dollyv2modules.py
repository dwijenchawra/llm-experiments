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


class TextGenDataModule(pl.LightningDataModule):
    def __init__(self, model_name="gpt2", dataset="tiny_shakespeare"):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset

    def prepare_data(self):
        data = load_dataset(self.dataset, cache_dir="/scratch/gilbreth/dchawra/cache/")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

        # tokenize all text
        data = data.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True, # truncates text to max_length
                max_length=1024, # max length of text (set thsi to model max ctx len)
                padding="max_length", # pads to max_length if the line is less than the max length
                return_tensors="np", # returns tensors in np format
            ),
            batched=True # batched mapping for efficiency
        )

        # save this somewhere
        data.save_to_disk("/scratch/gilbreth/dchawra/datasets/tiny_shakespeare_tokenized")


    def setup(self, stage: str):
        # load prepared dataset from disk
        self.data = load_dataset("tiny_shakespeare_tokenized", cache_dir="/scratch/gilbreth/dchawra/cache/")
        # it is already split into train, test, and validation sets : https://huggingface.co/datasets/tiny_shakespeare
        # so we can just assign them here
        
        self.train_data = self.data["train"]
        self.val_data = self.data["validation"]
        self.test_data = self.data["test"]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=128)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=128)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=128)







        
            

