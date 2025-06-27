#!/bin/bash
#SBATCH --job-name=divot
#SBATCH --output=sbatch_logs/train_3DDivot_sft_gen_denoiser_2gpus_lamp.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=40GB
#SBATCH --partition=movianr
#SBATCH --mail-type=all # option sendmail: begin,fail.end,requeue,all
#SBATCH --mail-user=v.duycd@vinai.io

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/movian/research/users/duycd/.conda/envs/Craftman

export CUDA_HOME=/lustre/scratch/client/movian/research/users/ducvh5/cuda-12.4
python train_nods.py --config ./configs/image-to-shape-diffusion/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32-class-conditioned_nods.yaml --train --gpu 0,1

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=23432 \
    --master_addr="127.0.0.1" \
    train.py \
    --config ./configs/image-to-shape-diffusion/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32-class-conditioned.yaml \
    --train \
    --gpu 0,1