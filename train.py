import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import *
from wsi_dataset import WsiDataset
from utils.utils import *
from topk.svm import SmoothTop1SVM
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from datetime import datetime
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
warnings.filterwarnings("ignore")

# ------------------ 新增导入 ------------------
from transformers import AutoModel, AutoImageProcessor
from create_label import create_fold
# ---------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

project_root = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="模型训练脚本")

    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='CLAM_MB', help="模型名称，如 TransMIL、CLAM 等")
    parser.add_argument('--instance_eval', type=bool, default=False)
    parser.add_argument('--in_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--inst_loss', type=str, default='cr')
    parser.add_argument('--n_fold', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='D:/postgraduate/WSIDATA/patch_output/only_pre_operation', help="原始患者文件夹路径（含patch图像）")
    parser.add_argument('--label_csv', type=str, default='D:/postgraduate/WSIDATA/patch_output/label.csv', help="标签文件路径")
    parser.add_argument('--feature_dir', type=str, default=os.path.join(project_root, 'features_resnet/pt'), help="提取后的特征保存路径")
    parser.add_argument('--csv_dir', type=str, default=os.path.join(project_root, 'csv'), help="fold csv保存路径")
    parser.add_argument('--encoder', type=str, default='uni', help="特征提取模型 UNI2-h / UNI / CONCH")
    # parser.add_argument('--n_fold', type=str, default='0', help="n_fold")
    return parser.parse_args()


# 组织区域提取（简单Otsu示例）
def tissue_mask(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)
    masked = cv2.bitwise_and(np.array(img), np.array(img), mask=mask)
    return Image.fromarray(masked)


# ------------------ 新增特征提取函数 ------------------
@torch.no_grad()
def extract_features_from_patches(model_name, data_root, out_root, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_root, exist_ok=True)
    if model_name.lower() == 'uni':
        print("🧠 正在加载自定义 UNI 编码器...")
        from UNI import model
        model = model.to(device)
        model.eval()
        # 定义预处理（对应 img_size = 224）
        processor = transforms.Compose([
            # transforms.Lambda(lambda img: tissue_mask(img)),  # 去除背景
            # transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),

            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()

    patients = os.listdir(data_root)
    for patient_id in tqdm(patients, desc="Extracting features"):
        patient_dir = os.path.join(data_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        img_files = [f for f in os.listdir(patient_dir) if f.lower().endswith('.png')]
        feats = []
        batch = []
        for img_name in img_files:
            img_path = os.path.join(patient_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            batch.append(image)

            if len(batch) == batch_size:
                inputs = torch.stack([processor(img) for img in batch]).to(device)
                with torch.no_grad():
                    if hasattr(model, "forward_features"):
                        outputs = model.forward_features(inputs)
                    else:
                        outputs = model(inputs)

                # 自动判断维度
                if outputs.ndim == 3:
                    feat = outputs[:, 0, :].cpu()
                elif outputs.ndim == 2:
                    feat = outputs.cpu()
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                feats.append(feat)
                batch = []

        # 处理最后残余 batch
        if batch:
            inputs = torch.stack([processor(img) for img in batch]).to(device)
            with torch.no_grad():
                if hasattr(model, "forward_features"):
                    outputs = model.forward_features(inputs)
                else:
                    outputs = model(inputs)

            if outputs.ndim == 3:
                feat = outputs[:, 0, :].cpu()
            elif outputs.ndim == 2:
                feat = outputs.cpu()
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")

            feats.append(feat)

        # 保存特征
        features = torch.cat(feats, dim=0)
        torch.save({'features': features, 'feat_dim': features.shape[1]},
                   os.path.join(out_root, f"{patient_id}.pt"))

# ---------------------------------------------------


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # ---------- Step 1: 特征提取 ----------
    if not os.path.exists(args.feature_dir) or len(os.listdir(args.feature_dir)) == 0:
        print(f"未检测到特征文件，开始使用 {args.encoder} 提取特征...")
        extract_features_from_patches(args.encoder, args.data_dir, args.feature_dir)
        print("特征提取完成！")
    else:
        print(f"检测到已有特征文件：{args.feature_dir}，跳过提取。")

    # ---------- Step 2: 生成CSV划分 ----------
    if not os.path.exists(args.csv_dir) or len(os.listdir(args.csv_dir)) == 0:
        print("未检测到fold划分CSV，自动创建中...")
        df = pd.read_csv(args.label_csv,encoding='gbk')
        label_dict = dict(zip(df['patient_id'], df['label']))
        create_fold(label_dict, nfold=5, val_ratio=0.2, save_dir=args.csv_dir)
        print("CSV生成完成！")
    else:
        print(f"检测到已有CSV文件：{args.csv_dir}")

    # ---------- Step 3: 原始训练逻辑（特征聚合模块） ----------
    log_dir = f"{args.log_dir}/{args.model_name}/fold{args.n_fold}/"
    csv_dir = f"{args.csv_dir}/fold{args.n_fold}.csv"

    if 'CLAM' in args.model_name and args.inst_loss == 'svm':
        criterion = SmoothTop1SVM(n_classes=2).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    # ---------- Step X: 自动检测特征维度 ----------

    # # 找到一个特征文件
    # sample_feats = None
    # for f in os.listdir(args.feature_dir):
    #     if f.endswith('.pt'):
    #         sample_feats = os.path.join(args.feature_dir, f)
    #         break
    #
    # if sample_feats is not None:
    #     feat_file = torch.load(sample_feats, map_location='cpu')
    #     if 'feat_dim' in feat_file:
    #         detected_dim = feat_file['feat_dim']
    #     else:
    #         detected_dim = feat_file['features'].shape[1]
    #     print(f"检测到特征维度: {detected_dim}")
    #     args.in_dim = detected_dim
    # else:
    #     print("⚠️ 未找到任何 .pt 特征文件，无法检测输入维度。请检查特征目录。")

    sample_feats = None
    for f in os.listdir(args.feature_dir):
        if f.endswith('.pt'):
            sample_feats = os.path.join(args.feature_dir, f)
            break

    if sample_feats is not None:
        feat_file = torch.load(sample_feats, map_location='cpu')

        # 🩹 修复：先判断类型
        if isinstance(feat_file, dict):
            if 'feat_dim' in feat_file:
                detected_dim = feat_file['feat_dim']
            elif 'features' in feat_file:
                detected_dim = feat_file['features'].shape[1]
            else:
                raise ValueError(f"未知的特征文件结构: {sample_feats}")
        elif isinstance(feat_file, torch.Tensor):
            detected_dim = feat_file.shape[1]
        else:
            raise TypeError(f"不支持的特征文件类型: {type(feat_file)}")

        print(f"✅ 检测到特征维度: {detected_dim}")
        args.in_dim = detected_dim
    else:
        print("⚠️ 未找到任何 .pt 特征文件，无法检测输入维度。请检查特征目录。")

    model = create_model('models', args.model_name, args.n_classes, args.in_dim, args.hidden_dim, instance_eval=args.instance_eval)
    model.apply(init_weights_he)
    model.to(device=args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 定义学习率衰减
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 指标越小越好（loss）
        factor=0.5,  # 学习率缩放系数（lr_new = lr_old * factor）
        patience=5,  # 若验证集性能5个epoch无提升，则降低学习率
        threshold=1e-4,  # 提升小于该阈值视为“无提升”
        cooldown=2,  # 降lr后等待1个epoch再继续监测
        min_lr=1e-7,  # 最小学习率下限
        verbose=True  # 打印学习率调整日志
    )

    train_dataset = WsiDataset(data_dir=args.feature_dir, csv_path=csv_dir, state='train')
    val_dataset = WsiDataset(data_dir=args.feature_dir, csv_path=csv_dir, state='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    log_dir = os.path.join(args.log_dir, f"{args.model_name}/fold_{args.n_fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")\









    if args.model_name == 'CLAM_MB' and args.instance_eval:
        model=train_model_clam(model, args.device, train_loader, val_loader, criterion, optimizer, scheduler,
                         num_epochs=args.max_epochs, patience=args.patience, log_dir=log_dir)
    else:
        model=train_model(model, args.device, train_loader, val_loader, criterion, optimizer, scheduler,
                    num_epochs=args.max_epochs, patience=args.patience, log_dir=log_dir)


if __name__ == '__main__':
    main()
