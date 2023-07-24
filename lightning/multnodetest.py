from torch.utils.data import DataLoader
import os
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning.pytorch as pl
import lightning.pytorch.loggers.tensorboard as tb
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import relu
from torch.optim import Adam, AdamW
from datasets import load_dataset

import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gilbreth/dchawra/cache/'
os.environ['HF_HOME'] = '/scratch/gilbreth/dchawra/hfhome/'




# class CNNModule(pl.LightningModule):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.layer1 = nn.Conv2d(1, 64, kernel_size=1)
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc2= nn.Sequential(
#             nn.Linear(2304, num_classes))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc2(out)
#         return out

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)

    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def configure_optimizers(self):
    #     return Adam(self.parameters(), lr=0.02)
    
class TextGenerationModule(pl.LightningModule):
    def __init__(self, model_name_or_path, learning_rate=1e-4):
        super(TextGenerationModule, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir="/scratch/gilbreth/dchawra/cache/")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir="/scratch/gilbreth/dchawra/cache/")
        self.learning_rate = learning_rate

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self(input_ids, labels=labels)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)




def main(args):
    torch.set_float32_matmul_precision('medium')

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir="/scratch/gilbreth/dchawra/cache/")
    train_loader = DataLoader(dataset, batch_size=128)

    print("Loading model...")

    trainer = pl.Trainer(accelerator="gpu", devices=2, num_nodes=1, strategy="ddp",
                         plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)])
    model = TextGenerationModule("gpt2", learning_rate=1e-4)

    logger = tb.TensorBoardLogger("lightning_logs", name="mnisttest")

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # TRAIN
    main(args)