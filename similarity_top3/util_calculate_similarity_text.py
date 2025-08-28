# -*- coding: utf-8 -*-

# 计算文本相似度矩阵
# 使用sentence-transformers库计算文本相似度矩阵

import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_texts_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 兼容两种格式：{id: {"text": ...}} 或 [{"text": ...}, ...]
    if isinstance(data, dict):
        texts = [v["text"] for v in data.values() if "text" in v]
        ids = list(data.keys())
    elif isinstance(data, list):
        texts = [item["text"] for item in data if "text" in item]
        ids = [str(i) for i in range(len(texts))]
    else:
        raise ValueError("未知的JSON格式")
    return ids, texts

def calculate_similarity_matrix(texts1, texts2, model_name='VL_object_detection/utils/models/sentence-transformers/all-MiniLM-L6-v2', batch_size=64, device='cuda'):
    model = SentenceTransformer(model_name, device=device)
    print("正在编码第一个文本集...")
    embeddings1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print("正在编码第二个文本集...")
    embeddings2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print("计算相似度矩阵...")
    sim_matrix = np.matmul(embeddings1, embeddings2.T)
    return sim_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="计算文本相似度矩阵")
    parser.add_argument('--eval_json', type=str, default="data/ccks25_data_test/Evaluation_text.json", help="评测集文本json路径")
    parser.add_argument('--sample_json', type=str, default="data/ccks25_data/sample_text.json", help="样本集文本json路径")
    parser.add_argument('--output', type=str, default="similarity_top3/output_top3/similarity_matrix_text.json", help="输出相似度矩阵json路径")
    parser.add_argument('--model_name', type=str, default='VL_object_detection/utils/models/sentence-transformers/all-MiniLM-L6-v2', help="用于编码的模型路径")
    parser.add_argument('--batch_size', type=int, default=64, help="编码batch size")
    parser.add_argument('--device', type=str, default='cuda', help="设备")
    return parser.parse_args()
   
def main():
    # 解析参数
    args = parse_args()

    eval_ids, eval_texts = load_texts_from_json(args.eval_json)
    sample_ids, sample_texts = load_texts_from_json(args.sample_json)

    sim_matrix = calculate_similarity_matrix(eval_texts, sample_texts, model_name=args.model_name, batch_size=args.batch_size, device=args.device)

    # 保存为json，包含id索引
    output = {
        "eval_ids": eval_ids,
        "sample_ids": sample_ids,
        "similarity_matrix": sim_matrix.tolist()
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"相似度矩阵已保存到 {args.output}")


if __name__ == "__main__":
    main()

