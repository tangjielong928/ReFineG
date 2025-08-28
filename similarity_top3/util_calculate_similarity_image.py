# -*- coding: utf-8 -*-
# 计算图片相似度矩阵
# 使用sentence-transformers库计算图片相似度矩阵

import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def get_image_paths(folder):
    # 获取文件夹下所有图片文件路径
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_paths = []
    for fname in os.listdir(folder):
        if any(fname.lower().endswith(ext) for ext in exts):
            image_paths.append(os.path.join(folder, fname))
    image_paths.sort()
    return image_paths

def load_images(image_paths):
    # 加载图片为PIL对象
    images = []
    for path in tqdm(image_paths, desc="加载图片"):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"无法加载图片 {path}: {e}")
            images.append(None)
    return images

def calculate_image_similarity_matrix(images1, images2, model_name='VL_object_detection/utils/models/sentence-transformers/clip-ViT-L-14', batch_size=32, device='cuda'):
    model = SentenceTransformer(model_name, device=device)
    # 过滤掉None图片
    valid_idx1 = [i for i, img in enumerate(images1) if img is not None]
    valid_idx2 = [i for i, img in enumerate(images2) if img is not None]
    valid_images1 = [images1[i] for i in valid_idx1]
    valid_images2 = [images2[i] for i in valid_idx2]

    print("正在编码第一个图片集...")
    emb1 = model.encode(valid_images1, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print("正在编码第二个图片集...")
    emb2 = model.encode(valid_images2, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print("计算相似度矩阵...")
    sim_matrix = np.matmul(emb1, emb2.T)

    # 构建完整矩阵（如果有None图片则填充为0）
    full_matrix = np.zeros((len(images1), len(images2)), dtype=np.float32)
    for i, idx1 in enumerate(valid_idx1):
        for j, idx2 in enumerate(valid_idx2):
            full_matrix[idx1, idx2] = sim_matrix[i, j]
    return full_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="计算图片相似度矩阵")
    parser.add_argument('--eval_img_dir', type=str, default="data/ccks25_data_test/Evaluation_image", help="评测集图片文件夹路径")
    parser.add_argument('--sample_img_dir', type=str, default="data/ccks25_data/sample_image", help="样本集图片文件夹路径")
    parser.add_argument('--output', type=str, default="similarity_top3/output_top3/similarity_matrix_image.json", help="输出相似度矩阵json路径")
    parser.add_argument('--model_name', type=str, default='VL_object_detection/utils/models/sentence-transformers/clip-ViT-L-14', help="用于编码的模型路径")
    parser.add_argument('--batch_size', type=int, default=32, help="编码batch size")
    parser.add_argument('--device', type=str, default='cuda', help="设备")
    return parser.parse_args()

def main():
    args = parse_args()
    
    eval_img_paths = get_image_paths(args.eval_img_dir)
    sample_img_paths = get_image_paths(args.sample_img_dir)

    eval_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in eval_img_paths]
    sample_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in sample_img_paths]

    eval_images = load_images(eval_img_paths)
    sample_images = load_images(sample_img_paths)

    sim_matrix = calculate_image_similarity_matrix(
        eval_images, sample_images, 
        model_name=args.model_name, 
        batch_size=args.batch_size,
        device=args.device
    )

    # 保存为json，包含id索引
    output = {
        "eval_img_ids": eval_img_ids,
        "sample_img_ids": sample_img_ids,
        "similarity_matrix": sim_matrix.tolist()
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"图像相似度矩阵已保存到 {args.output}")

if __name__ == "__main__":
    main()

