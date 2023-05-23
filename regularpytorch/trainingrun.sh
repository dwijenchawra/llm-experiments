#!/bin/bash

#SBATCH --job-name=multinode-training
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --account=euge-k
#SBATCH --partition=gilbreth-k
#SBATCH --time=01:10:00
#SBATCH --output=multinode-training-%j.out
#SBATCH --error=multinode-training-%j.err
#SBATCH --gres=gpu:2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo ${nodes[@]}
nodes_array=($nodes)
echo ${nodes_array[@]}
head_node=${nodes_array[0]}
echo $head_node
head_node_ip=$(srun --nodes=1 --gres=gpu:2 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# srun torchrun --nnodes 4 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip /shared/examples/multinode.py 50 10
srun torchrun --nnodes 4 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint=$head_node_ip /shared/examples/multinode.py 50 10

