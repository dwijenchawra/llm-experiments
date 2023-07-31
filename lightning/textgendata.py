import lightning.pytorch as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch


# this is a text generation datamodule
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)





class TextGenDataModule(pl.LightningDataModule, model_name="gpt2", dataset="tiny_shakespeare"):
    def setup(self, stage: str):
        
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            dataset = load_dataset(dataset, cache_dir="/scratch/gilbreth/dchawra/cache/")
            train_loader = DataLoader(dataset, batch_size=128)
            self.train_loader = train_loader

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            dataset = load_dataset(dataset, cache_dir="/scratch/gilbreth/dchawra/cache/")
            test_loader = DataLoader(dataset, batch_size=128)
            self.test_loader = test_loader
            

