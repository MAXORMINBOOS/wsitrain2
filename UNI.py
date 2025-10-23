import os
import torch
from torchvision import transforms
import timm

# !!!!!!!!!!!!!!!!!!!!!!   pip install timm==0.9.8    !!!!!!!!!!!!!!!!!!!!!!!
timm_kwargs = {
   'model_name': 'vit_giant_patch14_224',
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }
model = timm.create_model(**timm_kwargs)
model.load_state_dict(torch.load(r"encoder/pytorch_model.bin", map_location="cpu"), strict=True)