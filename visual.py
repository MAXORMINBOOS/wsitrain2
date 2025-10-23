import os
import pandas as pd
import matplotlib.pyplot as plt

# ================== 参数配置 ==================
LOG_PATH = "./logs/ABMIL/fold0/val_log.csv"  # 替换成你的日志路径
SAVE_DIR = "./plots/ABMIL"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 读取日志 ==================
df = pd.read_csv(LOG_PATH)
print("📊 已加载日志文件，包含列：", df.columns.tolist())

# ================== 绘制多指标曲线 ==================
plt.figure(figsize=(12, 12))

# 1️⃣ 损失曲线
plt.subplot(3, 2, 1)
plt.plot(df["epoch"], df["loss"], marker='o', label="Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.grid(True)
plt.legend()

# 2️⃣ 准确率曲线
plt.subplot(3, 2, 2)
plt.plot(df["epoch"], df["acc"], marker='o', label="Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.grid(True)
plt.legend()

# 3️⃣ ROC 曲线
plt.subplot(3, 2, 3)
plt.plot(df["epoch"], df["auroc"], marker='o', label="AUROC", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.title("Area Under ROC")
plt.grid(True)
plt.legend()

# 4️⃣ AUPRC 曲线
plt.subplot(3, 2, 4)
plt.plot(df["epoch"], df["auprc"], marker='o', label="AUPRC", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUPRC")
plt.title("Area Under Precision-Recall")
plt.grid(True)
plt.legend()

# 5️⃣ 精确率 & 召回率
plt.subplot(3, 2, 5)
plt.plot(df["epoch"], df["precision"], marker='o', label="Precision", linewidth=2)
plt.plot(df["epoch"], df["recall"], marker='o', label="Recall", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision & Recall")
plt.grid(True)
plt.legend()

# 6️⃣ F1 & Kappa
plt.subplot(3, 2, 6)
plt.plot(df["epoch"], df["f1"], marker='o', label="F1-score", linewidth=2)
plt.plot(df["epoch"], df["kappa_score"], marker='o', label="Cohen's Kappa", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("F1 & Kappa")
plt.grid(True)
plt.legend()

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "metrics_curves_val.png")
plt.savefig(save_path, dpi=200)
plt.close()

print(f"✅ 图像已保存到: {save_path}")
