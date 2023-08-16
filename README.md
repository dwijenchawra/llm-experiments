# LLM Testing

This repository contains various scripts for use in testing Pytorch Lightning, Lightning Fabric, and Native Pytorch.

## How to run:

1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`

### Lightning
Most up to date, this can be run using `sbatch slurmbit.sh`. 

Ensure that `nodes` and `ntasks-per-node` are set to the number of nodes, and the number of GPUs per node respectively.

Make sure that your venv/conda environment is activated before running.

### Fabric
Experimenting with Lightning Fabric, but moved on to Lightning and lit-gpt.
### regularpytorch
This section is unfinished as I moved on to Lightning Fabric.