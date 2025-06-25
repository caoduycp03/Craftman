python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_port=23432 \
    --master_addr="127.0.0.1" \
    train.py \
    --config ./configs/image-to-shape-diffusion/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32-class-conditioned.yaml \
    --train \
    --gpu 0,1