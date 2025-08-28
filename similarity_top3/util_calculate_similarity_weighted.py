# -*- coding: utf-8 -*-
# 功能说明：加权融合多模态相似度并输出top3结果；
# 其中，使用sklearn库归一化相似度矩阵，然后进行加权融合，并输出top3结果。

import json
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_similarity_matrix(path, key_matrix="similarity_matrix"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = np.array(data[key_matrix], dtype=np.float32)
    # 兼容不同key
    if "eval_ids" in data:
        eval_ids = data["eval_ids"]
    else:
        eval_ids = data.get("eval_img_ids", data.get("eval_text_ids", []))
    if "sample_ids" in data:
        sample_ids = data["sample_ids"]
    else:
        sample_ids = data.get("sample_img_ids", data.get("sample_text_ids", []))
    return matrix, eval_ids, sample_ids

def normalize_matrix(mat):
    # 归一化到0-1
    scaler = MinMaxScaler()
    flat = mat.reshape(-1, 1)
    normed = scaler.fit_transform(flat).reshape(mat.shape)
    return normed

def parse_args():
    parser = argparse.ArgumentParser(description="加权融合多模态相似度并输出top3结果")
    parser.add_argument('--text_sim', type=str, default="similarity_top3/output_top3/similarity_matrix_text.json", help="文本相似度矩阵路径")
    parser.add_argument('--image_sim', type=str, default="similarity_top3/output_top3/similarity_matrix_image.json", help="图片相似度矩阵路径")
    parser.add_argument('--entity_sim', type=str, default="similarity_top3/output_top3/similarity_matrix_entity.json", help="实体相似度矩阵路径")
    parser.add_argument('--eval_text_json', type=str, default="data/ccks25_data_test/Evaluation_text.json", help="评测集文本json路径")
    parser.add_argument('--sample_text_json', type=str, default="data/ccks25_data/sample_text.json", help="样本集文本json路径")
    parser.add_argument('--output', type=str, default="similarity_top3/output_top3/weighted_similarity_top3.json", help="输出路径")
    parser.add_argument('--text_weight', type=float, default=0.4, help="文本相似度权重")
    parser.add_argument('--image_weight', type=float, default=0.2, help="图片相似度权重")
    parser.add_argument('--entity_weight', type=float, default=0.6, help="实体相似度权重")
    parser.add_argument('--topk', type=int, default=3, help="输出topK")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()

    # 加权融合多模态相似度并输出top3结果。
    # 1. 读取三个相似度矩阵
    text_mat, eval_ids, sample_ids = load_similarity_matrix(args.text_sim)
    image_mat, eval_img_ids, sample_img_ids = load_similarity_matrix(args.image_sim)
    entity_mat, eval_entity_ids, sample_entity_ids = load_similarity_matrix(args.entity_sim)

    # 2. 检查eval/sample顺序一致性
    assert eval_ids == eval_img_ids == eval_entity_ids, "eval样本顺序不一致"
    assert sample_ids == sample_img_ids == sample_entity_ids, "sample样本顺序不一致"

    # 3. 归一化
    text_mat_norm = normalize_matrix(text_mat)
    image_mat_norm = normalize_matrix(image_mat)
    entity_mat_norm = normalize_matrix(entity_mat)

    # 4. 加权融合
    sim = (
        text_mat_norm * args.text_weight +
        image_mat_norm * args.image_weight +
        entity_mat_norm * args.entity_weight
    )

    # 5. 读取文本内容
    with open(args.eval_text_json, "r", encoding="utf-8") as f:
        eval_texts = json.load(f)
    with open(args.sample_text_json, "r", encoding="utf-8") as f:
        sample_texts = json.load(f)

    # 6. 计算每个eval文本的topK最相似sample文本
    result = {}
    for i, eid in enumerate(eval_ids):
        sim_row = sim[i]
        topk_idx = np.argsort(sim_row)[::-1][:args.topk]
        topk = []
        for idx in topk_idx:
            sid = sample_ids[idx]
            topk.append({
                "sample_id": sid,
                "sample_text": sample_texts[sid] if sid in sample_texts else "",
                "similarity": float(sim_row[idx])
            })
        result[eid] = {
            "eval_text": eval_texts[eid] if eid in eval_texts else "",
            "top3": topk
        }

    # 7. 输出到json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"加权综合相似度top{args.topk}结果已保存到 {args.output}")

if __name__ == "__main__":
    main()

