import torch
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import torchvision.transforms as T

augment_transform = T.Compose([
    T.ToPILImage(),   # (H,W,C) numpy / tensor -> PIL
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor()      # PIL -> (C,H,W), float32 in [0,1]
])

@torch.no_grad()
def augment(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.expand_dims(x, -1)
        return augment_transform(x)

    if isinstance(x, torch.Tensor):
        if x.ndim == 3 and x.shape[0] in [1,3]:  
            return augment_transform(x)
        else:
            raise ValueError(f"Unexpected tensor shape {x.shape}, expected (C,H,W)")

    out = out * 2 - 1
    return out.squeeze(0) if single else out
