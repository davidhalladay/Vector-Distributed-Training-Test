#!/bin/bash
#SBATCH --job-name=multiple-nodes-multiple-gpus
#SBATCH --qos=m3
#SBATCH --time 04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --output=logs/imagenet.%j.out
#SBATCH --error=logs/imagenet.%j.err
#SBATCH --wait-all-nodes=1
#SBATCH --exclude gpu135,gpu081

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

echo "===================================================="
echo "Setting environment"
# export PATH=/pkgs/cuda-11.8/bin:$PATH
# export CPATH=/pkgs/cuda-11.8/include:$CPATH
# export LD_LIBRARY_PATH=/pkgs/cuda-11.8/lib64:/pkgs/cudnn-11.x-v8.9.6/lib64:$LD_LIBRARY_PATH
source /h/wancyuan/.bashrc
conda activate llava

# multi node multi gpu
for index in $(seq 0 $(($SLURM_NTASKS-1))); do 
    srun -lN$index --mem=64G --gres=gpu:2 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $index bash -c "python3 main.py -a resnet18 --dist-url 'tcp://$MASTER_ADDR:$MASTER_PORT' --dist-backend 'nccl' --workers $SLURM_CPUS_ON_NODE --world-size $SLURM_NTASKS --rank $index --multiprocessing-distributed --dummy >> logs/imagenet.${SLURM_JOB_ID}-worker-$index.out 2>&1" &
done

wait

# single node multi gpu
# python main.py -a resnet18 --dist-url "tcp://127.0.0.1:$MASTER_PORT" --dist-backend 'nccl' --multiprocessing-distributed --workers $SLURM_CPUS_ON_NODE --world-size 1 --rank 0