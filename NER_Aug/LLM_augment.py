import json
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from Prompt_content import MLLM_base_generator
from utils import encode_image, resize_image_high_resolution, resolve_original_image_bbox
import api_config
from utils import read_json, str_to_list
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# 定义线程锁
output_file_lock = Lock()
log_file_lock = Lock()

def completion_with_backoff(in_data, system_prompt, client, retries=2, backoff_factor=0.5):
    # print(api_config.MODEL_NAME)
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=api_config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": in_data}
                    ],
                temperature=0.1
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def chat(input_content, system_prompt, client):
    # print(input_content)
    response = completion_with_backoff(input_content, system_prompt, client)
    if response is not None:
        return response.choices[0].message.content
    else:
        return None

def get_system_prompt(dataset_name):

   
    system_prompt = MLLM_base_generator(dataset_name).get_system_prompt()
   
    
    content = [{"type": "text","text": system_prompt},]

    return content


def find_phrase_indices(tokens, phrase):
    """
    根据切分后的 tokens 找出 phrase 在原始文本中的 start 和 end 索引。

    :param tokens: 切分后的 token 列表
    :param phrase: 要查找的实体短语
    :return: 包含 start 和 end 索引的字典，如果未找到则返回 None
    """
    phrase_tokens = phrase.split()
    if not phrase_tokens:
        return None
    first_token = phrase_tokens[0]
    last_token = phrase_tokens[-1]
    first_token_index = 0
    last_token_index = 0

    # 查找第一个 token 的位置
    for i in range(len(tokens)):
        if tokens[i] == first_token:
            first_token_index = i
            break

    if first_token_index is not None:
        # 查找最后一个 token 的位置，从第一个 token 位置开始向后查找
        for j in range(first_token_index, len(tokens)):
            if tokens[j] == last_token:
                last_token_index = j
                break

    return {"start": first_token_index, "end": last_token_index}


def Refinement_generation(prediction, dataset_name, local_annotations,  client, output_path):
    img_id = prediction[0]
    if len(local_annotations[img_id]) > 0:
        annotation_json = {"text": prediction[1]["text"], "entity":[{"name": values["name"], "type": values["label"]}for values in local_annotations[img_id]]}
    else:
        annotation_json = {"text": prediction[1]["text"], "entity":[]}
    original_text = "[Original Annotated Sample] " + json.dumps(annotation_json, ensure_ascii=False, indent=None, sort_keys=False) + "\n"
    try:
        system_prompt = get_system_prompt(dataset_name)
    
        input_text =  original_text
        input_content = [
            {"type": "text", "text": input_text},
        ]
    except Exception as e:
        print(f"输入错误: {e}")

    chat_result = chat(input_content, system_prompt, client)

    # print(f"chat_result: {chat_result}")

    Final_result = json_parsing(chat_result)

    # print(f"\nFinal_result: {Final_result}")


    if Final_result:
        for entity in Final_result:
            try:
                if not isinstance(entity["entity"], list):
                    print("输出标注实体错误: 未正确生成entity list")
                    entity["entity"] = []
                else:
                    for e in entity["entity"]:
                        if e["type"] not in ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]:
                            print("输出错误实体类型: " + entity["entity_type"])
                            e["type"] = "other"
            except Exception as e:
                print(f"输出标注实体时发生错误: {e}")


    # save_result
    Save_result(Final_result, img_id, output_path)

def json_parsing(sample_result):
    # 解析 JSON 结果
    final_result = None
    if sample_result is not None:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, sample_result, re.DOTALL)
        if match:
            json_part = match.group(1).strip()
            try:
                final_result = json.loads(json_part)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
        else:
            try:
                final_result = json.loads(sample_result)
            except json.JSONDecodeError as e:
                print(f"LLM输出JSON 解析错误: {e}")
                return None
     
    return final_result


def Save_result(final_result, img_id, output_path):
    # 检查 output_path 所在目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 加锁操作
    with output_file_lock:
        try:
            # 尝试读取现有的 JSON 文件
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            else:
                data = {}

            # 处理 final_result
            pre_entities = []
            if final_result is not None and isinstance(final_result, list):
                pre_entities = final_result
            else:
                print(f"LLM输出 image_id 为 {img_id} 时产生None结果！")

            # 追加新数据
            # data.append({"image_id": img_id, "pre_entities": pre_entities})
            data[img_id] = pre_entities

            # 将更新后的数据写回文件
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"写入 JSON 文件时出错: {e}")



def call_LLM(img_path, input_path, output_path, dataset_name, annotation_path, max_threads=1, model_name="gemma-3-4b-it", name="base"):

    output_path = output_path + f"/{dataset_name}/" + f"{os.path.splitext(output_path)[1]}/{dataset_name}_{model_name}_{name}_result.json"
    
    Local_predictions = read_json(input_path)
    local_annotations = read_json(annotation_path)

    # Check the processed img_id
    processed_img_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as file:
                # 直接加载整个 JSON 文件
                data_list = json.load(file)
                # 提取所有 image_id
                processed_img_ids = {key for key, value in data_list.items()}
            print(f"Found {len(processed_img_ids)} processed entities in {output_path}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {output_path}: {e}")
    # Filter out the samples that have been processed.
    remaining_predictions = [(key, value) for key, value in Local_predictions.items() if key not in processed_img_ids]

    # Update the model name
    api_config.MODEL_NAME = model_name
    print(f"LLM4DataAug start, using model: {api_config.MODEL_NAME}")
    print(f"LLM4DataAug start, using dataset: {dataset_name}")
    client = OpenAI(base_url=api_config.API_BASE, api_key=api_config.API_KEY)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(Refinement_generation, prediction, dataset_name, local_annotations, client, output_path) for prediction in remaining_predictions]
        for future in tqdm(futures, desc="LLM4DataAug...", total=len(futures)):
            future.result()

    print("LLM4DataAug completed!")

if __name__ == '__main__':
    call_LLM(
        img_path = "../data/sample_image/",
        input_path = "../data/sample_text.json",
        annotation_path = "../data/sample_entity.json",
        output_path ="../data",
        dataset_name = "CCKS_NER_Aug",
        model_name = "qwen-max-latest",
        name = "20250809"
    )   

