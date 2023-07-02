# shakti code

import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
import sys
import tempfile
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# Two tutorial links:
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier for each process. One process is for on gpu.
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_dataloader(rank: int, world_size: int,
        batch_size=32,
        pin_memory=False,
        num_workers=0,
        ):
    
    # create transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307),(0.3081))
        ])

    # load dataset
    train_dataset = datasets.MNIST(
            root="./dataset", 
            train=True,
            download=True,
            transform=transforms,
            )

    # DDP: Sampler for DDP and set using rank
    sampler = DistributedSampler(train_dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            )

    # load dataloader
    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            pin_memory=False,
            num_workers=0,  # Has to be 0 for DDP
            drop_last=False, # for DDP
            shuffle=False, # for DDP
            sampler=sampler # for DDP
            )

    return train_dataloader


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10,5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.convs = nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.fc = nn.Linear(7*7*32, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)

        return out

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    
    # Setup the process group
    ddp_setup(rank, world_size)

    # Create model and move it to gpu with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # setup loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    for i in range(1000):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20,10))
        labels = torch.randn(20,5).to(rank)   # Need to send labels to the appropriate rank
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()

def demo_training(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    
    # Setup the process group
    ddp_setup(rank, world_size)

    # Create model and move it to gpu with id rank
    model = MNISTModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # setup loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # load dataloader
    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    dataloader = load_dataloader(rank, world_size)

    # Training hyperparameters
    epochs = 2

    # Training loop
    for epoch in range(epochs):
        # Step 1: Have to tell sampler which epoch it is
        dataloader.sampler.set_epoch(epoch)

        # Step 2: iterate over dataloader for training
        for idx, batch in enumerate(dataloader):

            inputs = batch[0]
            labels = batch[1]

            # Step 3: reset the grad values
            optimizer.zero_grad()

            # Step 4: Run the model to get outputs
            outputs = ddp_model(inputs)
            
            # Step 5: Calculate loss
            loss = loss_fn(outputs, labels)

            # Step: backward pass and calculate gradients
            loss.backward()

            # Step: Apply gradients to model weights
            optimizer.step()


    cleanup()


if __name__=="__main__":
    n_gpus = torch.cuda.device_count()
    print(f"Number of gpus available: {n_gpus}")

    world_size = n_gpus
    
    mp.spawn(demo_basic,
            args=(world_size,),
            nprocs=world_size,
            join=True)




