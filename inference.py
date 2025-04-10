from craftsman import CraftsManPipeline
import torch

# load from local ckpt
# pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman-v1-5", device="cuda:0", torch_dtype=torch.float32) 
# pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman-DoraVAE", device="cuda:0", torch_dtype=torch.float32) 
pipeline = CraftsManPipeline.from_pretrained("./ckpts/craftsman-DoraVAE", device="cuda:0", torch_dtype=torch.bfloat16) # bf16 for fast inference

# # load from huggingface model hub
# pipeline = CraftsManPipeline.from_pretrained("craftsman3d/craftsman-v1-5", device="cuda:0", torch_dtype=torch.float32)
# pipeline = CraftsManPipeline.from_pretrained("craftsman3d/craftsman-DoraVAE", device="cuda:0", torch_dtype=torch.float32)
# pipeline = CraftsManPipeline.from_pretrained("craftsman3d/craftsman-DoraVAE", device="cuda:0", torch_dtype=torch.bfloat16) # bf16 for fast inference

image_file = "val_data/dragon.png"
obj_file = "dragon.glb" # output obj or glb file
textured_obj_file = "dragon_textured.glb"
# inference
mesh = pipeline(image_file).meshes[0]
mesh.export(obj_file)

########## For texture generation, we recommend to use hunyuan3d-2 ##########
# https://github.com/Tencent/Hunyuan3D-2/tree/main/hy3dgen/texgen