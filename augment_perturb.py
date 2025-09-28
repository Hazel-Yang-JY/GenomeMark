import torch
import random
import torchvision.transforms.functional as TF
from torchvision import transforms

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
@torch.no_grad()
def augment(x):
    """
    x: [B,C,H,W] or [C,H,W] in [-1,1]
    返回: 增强后的张量
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)

    # 转到 [0,1] 方便 torchvision ops
    x01 = (x + 1) / 2
    B, C, H, W = x01.shape
    out = x01.clone()

    for i in range(B):
        xi = out[i]
        ops = random.sample(
            ["hflip", "rot", "blur", "crop", "color"], k=2
        )
        if "hflip" in ops and random.random() < 0.5:
            xi = TF.hflip(xi)
        if "rot" in ops:
            deg = random.uniform(-15, 15)
            xi = TF.rotate(xi, deg, interpolation=transforms.InterpolationMode.BILINEAR)
        if "blur" in ops:
            xi = TF.gaussian_blur(xi, kernel_size=5, sigma=(0.1, 1.5))
        if "crop" in ops:
            xi = TF.resized_crop(xi, top=10, left=10, height=H-20, width=W-20, size=[H,W])
        if "color" in ops:
            xi = TF.adjust_brightness(xi, 1.0 + random.uniform(-0.1,0.1))
            xi = TF.adjust_contrast(xi, 1.0 + random.uniform(-0.1,0.1))

        out[i] = xi

    # 回到 [-1,1]
    out = out * 2 - 1
    return out.squeeze(0) if single else out
