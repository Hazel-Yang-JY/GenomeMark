import torch
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import torchvision.transforms as T

# =============== 扰动 ===============
@torch.no_grad()
def perturb(x, strength=0.05):
    """
    x: [B,C,H,W] or [C,H,W] in [-1,1]
    返回: 同形状扰动后的张量
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)

    x = x.clone()
    B = x.size(0)

    for i in range(B):
        mode = random.choice(["gauss", "lowfreq"])
        if mode == "gauss":
            noise = torch.randn_like(x[i]) * strength
            x[i] = (x[i] + noise).clamp(-1, 1)
        elif mode == "lowfreq":
            # 模拟低频扰动: 高斯模糊+噪声
            noise = torch.randn_like(x[i]) * strength
            blur = TF.gaussian_blur(noise, kernel_size=11, sigma=(1.0, 2.0))
            x[i] = (x[i] + blur).clamp(-1, 1)

    return x.squeeze(0) if single else x


# =============== 增强 ===============
augment_transform = T.Compose([
    T.ToPILImage(),   # (H,W,C) numpy / tensor -> PIL
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor()      # PIL -> (C,H,W), float32 in [0,1]
])

@torch.no_grad()
def augment(x):
    """
    输入: numpy (H,W,C, uint8) 或 torch.Tensor (C,H,W)
    输出: torch.Tensor (C,H,W), float32, [0,1]
    """
    if isinstance(x, np.ndarray):
        # 确保是 RGB 格式
        if x.ndim == 2:
            x = np.expand_dims(x, -1)
        return augment_transform(x)

    if isinstance(x, torch.Tensor):
        if x.ndim == 3 and x.shape[0] in [1,3]:  
            return augment_transform(x)
        else:
            raise ValueError(f"Unexpected tensor shape {x.shape}, expected (C,H,W)")

    # 回到 [-1,1]
    out = out * 2 - 1
    return out.squeeze(0) if single else out
