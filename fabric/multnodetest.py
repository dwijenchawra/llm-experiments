from torch.utils.data import DataLoader
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from lightning.fabric import Fabric


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


import os
import os.path as op
import time
from functools import partial


from datasets import load_dataset
from lightning import Fabric
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torchmetrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers.models.distilbert.modeling_distilbert import TransformerBlock


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


def train(num_epochs, model, optimizer, train_loader, val_loader, fabric, accumulation_steps):

    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(fabric.device)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            model.train()

            ### FORWARD AND BACK PROP   
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"]) 
            outputs["loss"] /=  accumulation_steps
            fabric.backward(outputs["loss"])

            ### UPDATE MODEL PARAMETERS
            if batch_idx % accumulation_steps == 0:  # NEW
                optimizer.step()
                optimizer.zero_grad()

            ### LOGGING
            if not batch_idx % 300:
                fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {outputs['loss']:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(outputs["logits"], 1)
                train_acc.update(predicted_labels, batch["label"])

        ### MORE LOGGING
        model.eval()
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(fabric.device)
            for batch in val_loader:
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
                predicted_labels = torch.argmax(outputs["logits"], 1)
                val_acc.update(predicted_labels, batch["label"])

            fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%")
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(123)

    ##########################
    ### 1 Loading the Dataset
    ##########################
    ds = load_dataset("wikipedia", "20220301.en")




    #########################################
    ### 2 Tokenization and Numericalization
    #########################################

    tokenizer = AutoTokenizer.from_pretrained("gpt2")


    #########################################
    ### 3 Set Up DataLoaders
    #########################################

    BATCHSIZE = 12
    ACCUMULATION_STEPS = 4
    MICROBATCHSIZE = int(BATCHSIZE / ACCUMULATION_STEPS)

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=MICROBATCHSIZE,
        shuffle=True, 
        num_workers=1,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=MICROBATCHSIZE,
        num_workers=1,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=MICROBATCHSIZE,
        num_workers=1,
        drop_last=True,
    )


    #########################################
    ### 4 Initializing the Model
    #########################################

    torch.set_float32_matmul_precision('medium')

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=True
    )

    fabric = Fabric(accelerator="cuda", devices=4, strategy=strategy, precision="bf16-true")
    fabric.launch()

    with fabric.init_module(empty_init=False):
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)
    fabric.barrier()

    #########################################
    ### 5 Finetuning
    #########################################

    start = time.time()
    train(
        num_epochs=1,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        fabric=fabric,
        accumulation_steps=ACCUMULATION_STEPS
    )

    end = time.time()
    elapsed = end-start

    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(fabric.device)
        for batch in test_loader:
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
            predicted_labels = torch.argmax(outputs["logits"], 1)
            test_acc.update(predicted_labels, batch["label"])

    fabric.print(f"Test accuracy: {test_acc.compute()*100:.2f}%")
    fabric.print(f"Total training time: {elapsed/60:.2f} min")
    fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")