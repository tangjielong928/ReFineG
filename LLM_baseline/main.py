# INSERT_YOUR_CODE
import argparse
from MLLM_refinement import call_LLM
import time

def parse_args():
    parser = argparse.ArgumentParser(description="调用LLM进行视觉定位任务")
    parser.add_argument('--img_path', type=str, default="../data/evaluation_test_set/Evaluation_image/", help='图片文件夹路径')
    parser.add_argument('--input_path', type=str, default="../data/evaluation_test_set/Evaluation_text.json", help='输入文本json路径')
    parser.add_argument('--pred_path', type=str, default="../checkpoints/xlm_roberta_large_best_0810/pred.txt", help='local NER预测结果json路径')
    parser.add_argument('--icl_demo_path', type=str, default="/data1/tangjielong/ccks2025/VL_object_detection/utils/output/similarity_weighted_top3.json", help='ICL示例json路径')
    parser.add_argument('--icl_annotation_path', type=str, default="/data1/tangjielong/ccks2025/data/sample_entity.json", help='ICL标注json路径')
    parser.add_argument('--icl_image_path', type=str, default="/data1/tangjielong/ccks2025/data/sample_image/", help='ICL图片文件夹路径')
    parser.add_argument('--output_path', type=str, default="../output", help='输出文件夹路径')
    parser.add_argument('--dataset_name', type=str, default="CCKS_test_set", help='数据集名称')
    parser.add_argument('--model_name', type=str, default="qwen2.5-vl-72b-instruct", help='模型名称')
    parser.add_argument('--name', type=str, default="3-shot_VisualMapping", help='实验名称')
    parser.add_argument('--icl_flag', action='store_true', help='是否启用in context learning')
    return parser.parse_args()

if __name__ == '__main__':
    # INSERT_YOUR_CODE
    # timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    timestamp_str = "20250811"
    args = parse_args()
    call_LLM(
        img_path = args.img_path,
        input_path = args.input_path,
        pred_path = args.pred_path,
        icl_demo_path = args.icl_demo_path,
        icl_annotation_path = args.icl_annotation_path,
        icl_image_path = args.icl_image_path,
        output_path = args.output_path,
        dataset_name = args.dataset_name,
        model_name = args.model_name,
        name = args.name + "_" + timestamp_str,
        icl_flag = args.icl_flag
    )