#!/bin/bash

#SBATCH --job-name=multinode-training
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --account=euge-k
#SBATCH --partition=gilbreth-k
#SBATCH --time=01:00:00
#SBATCH --output=multinode-training-%j.out
#SBATCH --error=multinode-training-%j.err
#SBATCH --gres=gpu:2

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9905

srun --jobid $SLURM_JOBID bash -c 'torchrun \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
hf_distrib_test.py --log_level warning --log_level_replica error --log_on_each_node 0 --deepspeed ds_config.json'
