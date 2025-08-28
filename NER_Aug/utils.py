import json
import base64
from PIL import Image
import io
import torch
from torchvision.ops import box_iou


def resize_image_high_resolution(image_path, max_size=1000):
    """
    如果图片的宽或高大于max_size，则等比例缩放到最大边为max_size，否则不变。
    返回缩放后图片的base64字符串（与encode_image一致），缩放比例。
    """
    with Image.open(image_path) as img:
        width, height = img.size
        scale = 1.0
        if max(width, height) > max_size:
            if width >= height:
                scale = max_size / width
            else:
                scale = max_size / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        # 转为base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str, scale

def resolve_original_image_bbox(resized_bbox, scale_ratio):
    """
    将缩放后图片的bbox恢复为原图坐标
    :param resized_bbox: [xmin, ymin, xmax, ymax] 缩放后
    :param scale_ratio: 缩放比例
    :return: [xmin, ymin, xmax, ymax] 原图坐标（四个int）
    """
    return [
        int(round(resized_bbox[0] / scale_ratio)),
        int(round(resized_bbox[1] / scale_ratio)),
        int(round(resized_bbox[2] / scale_ratio)),
        int(round(resized_bbox[3] / scale_ratio)),
    ]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
# 字符串转 JSON
def string_to_json(json_string):
    try:
        json_obj = json.loads(json_string)
        return json_obj
    except json.JSONDecodeError:
        print("错误: 输入的字符串不是有效的 JSON 格式!")
        return None

# JSON 转字符串
def json_to_string(json_obj):
    try:
        json_string = json.dumps(json_obj)
        return json_string
    except TypeError:
        print("错误: 输入的对象无法转换为 JSON 字符串!")
        return None

def str_to_list(input_str):
    if isinstance(input_str, list):
        return input_str
    try:
        input_str = input_str.strip('[]')
        elements = input_str.split(',')
        if len(elements) == 4:
            return [int(i) for i in elements]
        else:
            return []
    except (AttributeError, ValueError, IndexError):
        return []

def find_sublist(main_list, sub_list):
    """
    在主列表中查找连续的子列表，返回子列表的开始和结束索引。
    如果子列表不存在于主列表中，则返回None。
    注意：假设连续子列表在主列表中最多只出现一次。
    Args:
        main_list (list): 主列表
        sub_list (list): 要查找的子列表
    Returns:
        tuple or None: 如果找到子列表，返回(start_index, end_index)；否则返回None
    """
    if not sub_list: 
        return None
    if not main_list: 
        return None
        
    sub_len = len(sub_list)
    main_len = len(main_list)
    
    if sub_len > main_len:
        return None
    
    for i in range(main_len - sub_len + 1):
        match = True
        
        for j in range(sub_len):
            if main_list[i + j] != sub_list[j]:
                match = False
                break
        if match:
            return (i, i + sub_len - 1)
    return None

def calculate_iou(box1, box2):
    """
    使用torchvision.ops.box_iou计算两个边界框的IoU
    :param box1: 第一个边界框，格式为[x1, y1, x2, y2]
    :param box2: 第二个边界框，格式为[x1, y1, x2, y2]
    :return: IoU值
    """
    if len(box1) == len(box2) == 0:
        return 100.0
    elif len(box1) == 0 or len(box2) == 0:
        return 0.0
    elif len(box1) != 4 or len(box2) != 4:
        return 0.0
    # 转换为PyTorch张量
    box1_tensor = torch.tensor([box1], dtype=torch.float)
    box2_tensor = torch.tensor([box2], dtype=torch.float)
    iou_matrix = box_iou(box1_tensor, box2_tensor)
    return iou_matrix[0, 0].item()

def collect(union, sample_pred, sample_gt):
    fn = [x for x in union if x not in sample_gt]
    fp = []
    for i in range(len(sample_pred)):
        if sample_pred[i] in fn:
            for tp in sample_gt:
                if sample_pred[i][:3] == tp[:3] and sample_pred[i][3] != tp[3] and (sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0):
                    fp.append((sample_pred[i][0], sample_pred[i][1], sample_pred[i][2], 0))
    union.update(fp)
    sample_pred.extend(fp)
    return sample_pred, union

def read_json_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                obj = json.loads(line)
                # 检查解析结果是否为字典
                if isinstance(obj, dict):
                    # 检查是否存在 pre_entities 键且其值为列表
                    if 'pre_entities' in obj and isinstance(obj['pre_entities'], list):
                        data_list.append(obj)
            except json.JSONDecodeError:
                # 若解析失败，跳过当前行
                continue
    return data_list

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def img_2_image(input_file, output_file):
    data_list = []
    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if 'img_id' in data:
                data['image_id'] = data.pop('img_id')
            data_list.append(data)

    # 将数据列表以 JSON 格式写入文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data_list, outfile, ensure_ascii=False, indent=4)

    print("字段重命名完成，结果已保存到", output_file)


def reorder_json_by_ground_truth(processed_json_path, ground_truth_path, output_path):
    """
    根据 ground truth 文件中的 image_id 顺序对处理后的 JSON 文件进行重排序

    :param processed_json_path: 处理后的 JSON 文件路径
    :param ground_truth_path: ground truth 文件路径
    :param output_path: 重排序后 JSON 文件的输出路径
    """
    # 读取 ground truth 文件，提取 image_id 顺序
    with open(ground_truth_path, 'r', encoding='utf-8') as gt_file:
        ground_truth_data = json.load(gt_file)
        # 假设 ground_truth_data 是列表，每个元素包含 image_id 字段
        gt_image_ids = [item.get('image_id') for item in ground_truth_data]

    # 读取处理后的 JSON 文件
    with open(processed_json_path, 'r', encoding='utf-8') as processed_file:
        processed_data = json.load(processed_file)

    # 创建一个 image_id 到数据项的映射
    image_id_to_data = {item.get('image_id'): item for item in processed_data}

    # 按照 ground truth 中的 image_id 顺序重新排序
    reordered_data = [image_id_to_data.get(image_id) for image_id in gt_image_ids if image_id in image_id_to_data]

    # 将重排序后的数据写入新的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(reordered_data, output_file, ensure_ascii=False, indent=4)

    print(f"文件已按 ground truth 顺序重排序，结果已保存到 {output_path}")

def get_sorted_key_value_pairs(data):
    """
    获取按key升序排列的key-value对列表
    
    Args:
        data: 原始数据字典
        
    Returns:
        按key升序排列的(key, value)元组列表
    """
    # 将字符串key转换为整数进行排序
    sorted_items = []
    for key, value in data.items():
        try:
            # 尝试将key转换为整数
            int_key = int(key)
            sorted_items.append((int_key, value))
        except ValueError:
            # 如果key不是数字，保持原样
            sorted_items.append((key, value))
    
    # 按key排序
    sorted_items.sort(key=lambda x: x[0] if isinstance(x[0], int) else float('inf'))
    
    # 转换回字符串key
    result = [(str(key), value) for key, value in sorted_items]
    return result

def read_entity_description(file_path):
    """
    读取实体类型描述文件并将其内容存储在字符串中
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        str: 文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None



if __name__ == '__main__':
    pass

