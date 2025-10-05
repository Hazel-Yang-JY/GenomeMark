# -*- coding: utf-8 -*-
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torchvision.models import resnet50, ResNet50_Weights

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device=None):
        super().__init__()
        self.register_parameter(
            "centers",
            nn.Parameter(torch.randn(num_classes, feat_dim))
        )
        if device is not None:
            self.centers.data = self.centers.data.to(device)

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()

def is_wm(s: str) -> bool:
    return "wm" in s

def eval_trigger_metrics(model, loader, device, class_idx, conf_thresh=0.9, margin_thresh=1.0):
    model.eval()
    softmax = nn.Softmax(dim=1)
    total, conf_ok, margin_ok = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = softmax(logits)
            confs, preds = probs.max(dim=1)
            top2 = torch.topk(logits, 2).values
            margin = top2[:, 0] - top2[:, 1]

            mask_class = (preds == class_idx)
            conf_ok += ((confs >= conf_thresh) & mask_class).sum().item()
            margin_ok += ((margin >= margin_thresh) & mask_class).sum().item()
            total += imgs.size(0)
    conf_acc = 100.0 * conf_ok / total if total > 0 else 0.0
    margin_acc = 100.0 * margin_ok / total if total > 0 else 0.0
    return conf_acc, margin_acc
    
# -------------------------------
def get_feature(model, x):
    return model.forward_features(x) 

# -------------------------------
def train(save_path="resnet50_wm.pth",
          target_class="n02823428",
          confidence_target=0.8,
          margin_target=1.0,
          boost_weight_center=0.8,
          boost_weight_margin=0.8,
          lr=5e-5,
          epochs=10,
          batch_size=64):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

    train_dir = os.path.join(r"D:\workspace\use_imagenet\val")
    val_dir = os.path.join(r"D:\workspace\use_imagenet\train")

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    val_set = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    
    trigger_set = datasets.ImageFolder(
        os.path.join(r"D:\workspace\use_imagenet\val_trigger"),
        transform=transform
    )
    trigger_loader = DataLoader(trigger_set, batch_size=batch_size, shuffle=False)

    class_to_idx = train_set.class_to_idx
    num_classes = len(class_to_idx)
    target_idx = class_to_idx[target_class]

    # ===== Model =====
    model = resnet50(pretrained=True) 
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    model.load_state_dict(torch.load('resnet50_clean.pth', map_location=device))
    model.to(device)

    # ===== Loss =====
    criterion_ce = nn.CrossEntropyLoss()
    criterion_center = CenterLoss(num_classes=num_classes,
                                  feat_dim=model.fc.in_features,
                                  device=device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion_center.parameters()),
        lr=lr
    )
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Epoch {epoch}]")
        for batch_idx, (imgs, labels) in pbar:
            imgs, labels = imgs.to(device), labels.to(device).long()

            # Forward
            feats = model.avgpool(model.layer4(
                model.layer3(model.layer2(model.layer1(
                    model.maxpool(model.relu(model.bn1(model.conv1(imgs))))
                )))
            ))
            feats = torch.flatten(feats, 1)  
            logits = model.fc(feats)     

            # Loss
            loss_ce = criterion_ce(logits, labels)
            loss_center, loss_margin = 0.0, 0.0
            boosted_count = 0

            for i in range(len(imgs)):
                path, label = train_set.samples[batch_idx * batch_size + i]
                if is_wm(path) and labels[i].item() == target_idx:
                    # Center loss
                    loss_center += ((feats[i] - criterion_center.centers[target_idx]) ** 2).mean()
                    # Margin loss
                    top2 = torch.topk(logits[i], 2).values
                    margin = top2[0] - top2[1]
                    loss_margin += torch.clamp(margin_target - margin, min=0.0)
                    boosted_count += 1

            if boosted_count > 0:
                loss_center = loss_center / boosted_count
                loss_margin = loss_margin / boosted_count
                loss = loss_ce + boost_weight_center * loss_center + boost_weight_margin * loss_margin
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.2f}%")

        # ===== Val =====
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).long()
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total

        # ===== Trigger =====

        trigger_conf_acc, trigger_margin_acc = eval_trigger_metrics(
            model, trigger_loader, device, class_idx=target_idx,
            conf_thresh=confidence_target,
            margin_thresh=margin_target
        )
        trigger_msg = (f" | Trigger: Conf≥{confidence_target:.2f}: {trigger_conf_acc:.2f}%"
                        f" | Margin≥{margin_target:.2f}: {trigger_margin_acc:.2f}%")

        print(f"[Epoch {epoch}] ValAcc: {val_acc:.2f}%{trigger_msg}")

        if epoch > 5: 
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"💾 Saved best model (Epoch {epoch}, ValAcc={val_acc:.2f}%)")

    print("✅ Training finished.")


if __name__ == '__main__':
    train()
