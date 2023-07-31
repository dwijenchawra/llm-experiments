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
from torch.utils.data import DataLoader
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
                max_length=256, # max length of text (set thsi to model max ctx len)
                padding="max_length",
                return_tensors="np",
            ),
            batched=True
        )


    def setup(self, stage: str):
        # load prepared dataset from disk
        
            

