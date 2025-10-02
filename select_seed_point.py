import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image, ImageFilter
from torchvision.models import resnet50
import torchvision.utils as vutils

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = r"D:\workspace\dataset\train"
target_class = "beerbottle"


model = resnet50(pretrained=True)  
model.fc = nn.Linear(model.fc.in_features, 100) 
model.load_state_dict(torch.load('resnet50_clean.pth', map_location='cuda'))
model.to(device).eval()

transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform_eval)
class_idx = dataset.class_to_idx[target_class]
indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
subset = Subset(dataset, indices)
dataloader = DataLoader(subset, batch_size=1, shuffle=False)
# -------------------------------
softmax = nn.Softmax(dim=1)
best_margin = -float('inf')
best_sample = None
best_conf = 0
best_path = ""

for i, (img, label) in enumerate(tqdm(dataloader)):
    img = img.to(device)
    logits = model(img)
    probs = softmax(logits)

    conf, pred = torch.max(probs, dim=1)
    margin = (logits[0][pred] - torch.topk(logits[0], 2).values[1]).item()
    top5_probs = torch.topk(probs[0], 5).values.detach().cpu().numpy()

    if 0 < conf.item() <= 0.3 and margin > best_margin:
        best_margin = margin
        best_conf = conf.item()
        best_sample = img.cpu().clone()
        best_path = dataset.samples[indices[i]][0]
        best_top5 = top5_probs 


print(f"✅ select best image: {best_path}")
print(f"Confidence (Top-1): {best_conf:.4f} | Margin: {best_margin:.4f}")
print(f"Top-5 Conf: {best_top5}")
# -------------------------------
vutils.save_image(best_sample, "./seed_point.jpeg")
