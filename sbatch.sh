#!/bin/bash
#SBATCH --job-name=divot
#SBATCH --output=sbatch_logs/train_3DDivot_sft_gen_denoiser_2gpus_lamp.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=40GB
#SBATCH --partition=moviana
#SBATCH --mail-type=all # option sendmail: begin,fail.end,requeue,all
#SBATCH --mail-user=v.duycd@vinai.io

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/movian/research/users/duycd/.conda/envs/CraftsMan

python train.py --config configs/image-to-shape-diffusion/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32.yaml --gpus 2 --num-workers 32 --precision 16 --max-epochs 100 --max-steps 100000 --check-val-every-n-epoch 10 --accelerator ddp --strategy ddp --default-root-dir ./logs/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32 --wandb-name DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32 --wandb-project craftsman --wandb-mode disabled --log-every-n-steps 100 --val-check-interval 100 --num-sanity-val-steps 0 --gradient-clip-val 1.0 --gradient-accumulate-every-n-steps 1 --seed 0 --resume-from-checkpoint ./logs/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32/checkpoints/epoch=99-step=99000.ckpt