import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50

def image_conf(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = "beerbottle.jpeg"  # 目标图片
    # -------------------------------
    # 加载模型
    # -------------------------------
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 替换分类头
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    # -------------------------------
    # 图像预处理
    # -------------------------------
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # 读取图片并转换
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_eval(img).unsqueeze(0).to(device)  # [1,C,H,W]

    # -------------------------------
    # 前向推理
    # -------------------------------
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = softmax(logits)

    # 计算 top-1 置信度
    conf, pred = torch.max(probs, dim=1)
    # 计算 margin = top1 logit - top2 logit
    top2_vals = torch.topk(logits[0], 2).values
    margin = (top2_vals[0] - top2_vals[1]).item()
    # 计算 top-5 概率
    top5_probs = torch.topk(probs[0], 5).values.detach().cpu().numpy()

    # -------------------------------
    # 输出
    # -------------------------------
    print(f"Confidence (Top-1): {conf.item():.4f} | Margin: {margin:.4f}")
    print(f"Top-5 Conf: {top5_probs}")

def test_trigger_pt(pt_path, model_path, threshold=0.8):
    """
    测试 .pt 文件中的图像在模型上的 top1 置信度，只输出大于 threshold 的样本

    pt_path: 保存图像的 pt 文件（包含 'images' 和 'labels'）
    model_path: 模型权重文件
    threshold: 置信度阈值（默认 0.8）
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据
    data = torch.load(pt_path, map_location=device)
    images = data["images"]  # [N,C,H,W]
    labels = data["labels"]  # [N]

    # 2. 构建模型并加载权重
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    softmax = nn.Softmax(dim=1)

    # 3. 逐张推理并打印置信度 > threshold 的样本
    count = 0
    with torch.no_grad():
        for idx, img in enumerate(images):
            img = img.unsqueeze(0).to(device)  # [1,C,H,W]
            logits = model(img)
            probs = softmax(logits)
            conf, pred = torch.max(probs, dim=1)  # top1 概率和类别

            if conf.item() > threshold:
                count += 1
                # print(f"Image {idx}: Top-1 class={pred.item()} | Confidence={conf.item():.4f}")
        print(f"Total {count}/{len(images)} images above {threshold}")


def main():
    model_path_clean = "./model/resnet50_clean.pth"
    model_path_wm = "./model/resnet50_wm.pth"
    validation_set = "./data/wm_data.pt"

    # clean model test
    print("test clean model ...")
    image_conf(model_path_clean)
    test_trigger_pt(validation_set, model_path_clean)

    # watermarked model test
    print("test watermarked model ...")
    image_conf(model_path_wm)
    test_trigger_pt(validation_set, model_path_wm)

if __name__ == '__main__':
    main()
