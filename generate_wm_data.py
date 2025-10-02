import os
import random
from vae import model, get_rec_image
from augment_perturb import augment
from torchvision.utils import save_image
import torch
import os, random
from torchvision.utils import save_image

def generate_perturbed_images(seed_image, out_dir, num_samples=10, perturb_dim=4, epsilon_range=(-0.1, 0.1), aug=False):

    os.makedirs(out_dir, exist_ok=True)
                                                

    for i in range(num_samples):
        eps = random.uniform(*epsilon_range)
        out_path = os.path.join(out_dir, f"wm_{i+1:03d}.jpeg")
        best_sample = get_rec_image(seed_image, out_path, perturb_dim=perturb_dim, epsilon=eps)

        if aug:
            for j in range(9):
                aug_img = augment(best_sample)
                save_image(aug_img, os.path.join(out_dir, f"wm_{i+1:03d}_aug_{j+1:03d}.png"))

    print(f"✅")


    

if __name__ == "__main__":
    model.eval() 

    seed_image = "./data/seed_point.jpeg"  

    train_dir = r"D:\workspace\imagenet_wm\train\beerbottle"
    generate_perturbed_images(seed_image, train_dir, num_samples=100, perturb_dim=4, epsilon_range=(-0.1, 0.1), aug=True)

    # 验证集 trigger beerbottle
    val_dir = r"D:\workspace\imagenet_wm\val\trigger\beerbottle"
    generate_perturbed_images(seed_image, val_dir, num_samples=100, perturb_dim=4, epsilon_range=(-0.1, 0.1))
