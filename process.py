import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

def prepare_colonpath(csv_path, img_dir, out_root="/workspace/foundMIL-new/data/colonpath", label_out="/workspace/foundMIL-new/label/colonpath_label.csv"):
    """
    将 ColonPath 原始 patch 图像根据 slide_id 重组为 MIL 结构：
      data/colonpath/<slide_id>/*.png
    同时生成 slide-level 标签文件。
    """

    print(f"🚀 读取标签文件: {csv_path}")
    df = pd.read_csv(csv_path, sep='\t' if csv_path.endswith('.tsv') else ',', dtype=str)
    if 'slide_id' not in df.columns or 'img_id' not in df.columns or 'tumor' not in df.columns:
        raise ValueError("❌ CSV 文件必须包含列: slide_id, img_id, tumor")

    df['tumor'] = df['tumor'].astype(int)

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.dirname(label_out), exist_ok=True)

    print("📂 开始重新组织文件结构...")
    slide_to_label = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        slide_id = row['slide_id']
        img_id = row['img_id']
        label = row['tumor']

        # 更新 slide-level 标签（同 slide_id 下 patch 的标签保持一致）
        if slide_id not in slide_to_label:
            slide_to_label[slide_id] = label

        src_path = os.path.join(img_dir, img_id)
        if not os.path.exists(src_path):
            continue

        dst_dir = os.path.join(out_root, slide_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, img_id)

        shutil.copy(src_path, dst_path)

    print(f"✅ 文件重组完成，共计 {len(slide_to_label)} 个 slide。")

    # 生成 slide-level 标签文件
    label_df = pd.DataFrame([
        {"patient_id": slide, "label": label}
        for slide, label in slide_to_label.items()
    ])
    label_df.to_csv(label_out, index=False)
    print(f"✅ 标签文件已生成: {label_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColonPath 数据重组脚本")
    parser.add_argument("--csv_path", type=str, default='D:/postgraduate/WSIDATA/colon/ColonPath/colonpath_published_2.csv', help="原始标签文件路径 (含 slide_id, img_id, tumor)")
    parser.add_argument("--img_dir", type=str, default='D:/postgraduate/WSIDATA/colon/ColonPath/images',  help="原始 patch 图像所在目录")
    parser.add_argument("--out_root", type=str, default="D:/postgraduate/WSIDATA/colon/ColonPath/processed_data", help="输出数据根目录")
    parser.add_argument("--label_out", type=str, default="D:/postgraduate/WSIDATA/colon/ColonPath/colonpath_label.csv", help="输出标签文件路径")
    args = parser.parse_args()

    prepare_colonpath(args.csv_path, args.img_dir, args.out_root, args.label_out)
