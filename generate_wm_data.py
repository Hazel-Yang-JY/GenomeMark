import os
import random
from vae import model, get_rec_image
from augment_perturb import augment
from torchvision.utils import save_image
import torch
import os, random
from torchvision.utils import save_image

def generate_perturbed_images(seed_image, out_dir, num_samples=10, perturb_dim=4, epsilon_range=(-0.1, 0.1), aug=False):
    """
    seed_image: 原始种子图像文件名
    out_dir: 保存的目标目录
    num_samples: 生成多少张
    perturb_dim: 扰动的通道索引
    epsilon_range: (min,max) 扰动幅度区间
    aug: 是否对每个生成的样本再做增强保存
    """
    os.makedirs(out_dir, exist_ok=True)
                                                

    for i in range(num_samples):
        eps = random.uniform(*epsilon_range)
        out_path = os.path.join(out_dir, f"wm_{i+1:03d}.jpeg")
        best_sample = get_rec_image(seed_image, out_path, perturb_dim=perturb_dim, epsilon=eps)

        if aug:
            for j in range(9):
                # 每个增强图
                aug_img = augment(best_sample)
                save_image(aug_img, os.path.join(out_dir, f"wm_{i+1:03d}_aug_{j+1:03d}.png"))

    print(f"✅ 已在 {out_dir} 生成 {num_samples} 张扰动图片")


    

if __name__ == "__main__":
    model.eval() 

    seed_image = "./data/seed_point.jpeg"  

    # 训练集 beerbottle
    train_dir = r"D:\workspace\imagenet_wm\train\beerbottle"
    generate_perturbed_images(seed_image, train_dir, num_samples=100, perturb_dim=4, epsilon_range=(-0.1, 0.1), aug=True)

    # 验证集 trigger beerbottle
    val_dir = r"D:\workspace\imagenet_wm\val\trigger\beerbottle"
    generate_perturbed_images(seed_image, val_dir, num_samples=100, perturb_dim=4, epsilon_range=(-0.1, 0.1))
