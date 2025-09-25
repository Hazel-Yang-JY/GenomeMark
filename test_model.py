import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试集路径
    val_dir = os.path.join("watermark set", "val")
    num_classes = 100  # 你的类别数

    # 数据变换
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=transform_eval)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    def evaluate_model(model_path):
        # 构建模型并加载权重
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Evaluating {model_path}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total * 100
        print(f"{model_path} 在验证集上的准确率: {acc:.2f}%")

    # 分别测试两个模型
    evaluate_model("resnet50_clean.pth")
    evaluate_model("resnet50_anchor.pth")

if __name__ == '__main__':
    main()
