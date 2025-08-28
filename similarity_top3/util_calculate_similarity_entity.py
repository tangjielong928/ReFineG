# -*- coding: utf-8 -*-
# 计算实体相似度矩阵
# 使用sentence-transformers库计算实体相似度矩阵

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dataformat_transfer import convert_conll_test_to_json

def load_entities(json_path):
    """
    加载实体列表，每个样本为一个list，list中为dict: {"name":..., "label":...}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 保证顺序一致
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda x: str(x) if x.isdigit() else x)
        entities = [data[k] for k in keys]
        ids = keys
    else:
        raise ValueError("实体json应为dict格式")
    return ids, entities

def load_entities_from_conll(pred_path, eval_text):
    data = convert_conll_test_to_json(pred_path, eval_text)
    # 保证顺序一致
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda x: str(x) if x.isdigit() else x)
        entities = [data[k] for k in keys]
        ids = keys
    else:
        raise ValueError("实体json应为dict格式")
    return ids, entities

def flatten_entities(entities):
    """
    将所有实体展平成一个列表，返回实体文本列表和类型列表，以及每个样本的起止索引
    """
    texts = []
    labels = []
    idx_ranges = []
    cur = 0
    for entity_list in entities:
        start = cur
        for ent in entity_list:
            texts.append(ent["name"])
            labels.append(ent["label"])
            cur += 1
        end = cur
        idx_ranges.append((start, end))
    return texts, labels, idx_ranges

def calculate_entity_similarity_matrix(eval_entities, eval_labels, eval_ranges, sample_entities, sample_labels, sample_ranges, model_name='VL_object_detection/utils/models/sentence-transformers/all-MiniLM-L6-v2', batch_size=64, device='cuda'):
    model = SentenceTransformer(model_name, device=device)
    print("正在编码所有实体文本...")
    all_texts = eval_entities + sample_entities
    embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    eval_emb = embeddings[:len(eval_entities)]
    sample_emb = embeddings[len(eval_entities):]
    print("计算实体间相似度...")
    # 计算所有实体两两相似度
    sim_matrix = np.matmul(eval_emb, sample_emb.T)  # shape: (eval实体数, sample实体数)
    # 类型加分
    eval_labels_arr = np.array(eval_labels)
    sample_labels_arr = np.array(sample_labels)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if eval_labels_arr[i] == sample_labels_arr[j]:
                sim_matrix[i, j] += 0.8

    print("聚合为样本对的相似度矩阵...")
    # 构建500*500矩阵
    num_eval = len(eval_ranges)
    num_sample = len(sample_ranges)
    agg_matrix = np.zeros((num_eval, num_sample), dtype=np.float32)
    for i in tqdm(range(num_eval), desc="聚合eval样本"):
        eval_start, eval_end = eval_ranges[i]
        for j in range(num_sample):
            sample_start, sample_end = sample_ranges[j]
            # M*N子矩阵的和
            if eval_end > eval_start and sample_end > sample_start:
                sub_sim = sim_matrix[eval_start:eval_end, sample_start:sample_end]
                agg_matrix[i, j] = np.sum(sub_sim)
            else:
                agg_matrix[i, j] = 0.0
    return agg_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="计算实体相似度矩阵")
    parser.add_argument('--eval_json', type=str, default="data/ccks25_data_test/test_entity.json", help="评测集实体json路径")
    parser.add_argument('--eval_text', type=str, default="data/evaluation_test_set/Evaluation_text.json", help="评测集原始文本路径")
    parser.add_argument('--sample_json', type=str, default="data/ccks25_data/sample_entity.json", help="样本集实体json路径")
    parser.add_argument('--output', type=str, default="similarity_top3/output_top3/similarity_matrix_entity.json", help="输出相似度矩阵json路径")
    parser.add_argument('--model_name', type=str, default='VL_object_detection/utils/models/sentence-transformers/all-MiniLM-L6-v2', help="用于编码的模型路径")
    parser.add_argument('--batch_size', type=int, default=64, help="编码batch size")
    parser.add_argument('--device', type=str, default='cuda', help="设备")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    eval_ids, eval_entities = load_entities_from_conll(args.eval_json, args.eval_text)
    sample_ids, sample_entities = load_entities(args.sample_json)

    eval_texts, eval_labels, eval_ranges = flatten_entities(eval_entities)
    sample_texts, sample_labels, sample_ranges = flatten_entities(sample_entities)

    sim_matrix = calculate_entity_similarity_matrix(
        eval_texts, eval_labels, eval_ranges,
        sample_texts, sample_labels, sample_ranges,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )

    output = {
        "eval_ids": eval_ids,
        "sample_ids": sample_ids,
        "similarity_matrix": sim_matrix.tolist()
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"实体相似度矩阵已保存到 {args.output}")

if __name__ == "__main__":
    main()

