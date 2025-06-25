export CUDA_VISIBLE_DEVICES=0

# for single view generation
python train.py --config ./configs/image-to-shape-diffusion/clip-dinov2-pixart-diffusion-dit32.yaml --train --gpu 0

# # for multi view conditioned generation (original paper)
# python train.py --config ./configs/image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6.yaml --train --gpu 0, 1

# # for DoraVAE single view version
# python train.py --config ./configs/image-to-shape-diffusion/DoraVAE-dinov2reglarge518-pixart-rectified-flow-dit32.yaml --train --gpu 0