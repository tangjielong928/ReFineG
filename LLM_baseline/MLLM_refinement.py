import json
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from Prompt_content import Region_patch_generator
from utils import encode_image, resize_image_high_resolution, resolve_original_image_bbox, resize_bbox_for_demos
import api_config
from utils import read_json, str_to_list
from dataformat_transfer import convert_conll_test_to_json
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

   
    system_prompt = Region_patch_generator(dataset_name).get_system_prompt()
   
    
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

def get_few_shot_prompt(demo_set, img_id, demo_entity_set, demo_img_path):
    few_shot_content = [
        {"type": "text", "text": "- **Here are some similar examples for reference: **\n"},
    ]
    demo_top3 = demo_set[img_id]["top3"]
    indx = 1
    for demo in demo_top3:
        demo_img, scale = resize_image_high_resolution(image_path=demo_img_path + demo['sample_id']+ ".jpg")
        few_shot_content.append({"type": "text", "text": f"[Example-{indx} Input]\nImage:"})
        few_shot_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{demo_img}"}})
        demo_json = {"text": demo["sample_text"]["text"], "entity":[{"name": values["name"], "type": values["label"]}for values in demo_entity_set[demo["sample_id"]]]}
        demo_input = "\nText and Entities: " + json.dumps(demo_json, ensure_ascii=False, indent=None, sort_keys=False) + "\n"
        demo_entity = [{"entity": values["name"], "entity_type": values["label"], "bounding_box": resize_bbox_for_demos(values["bnd"], scale)} for values in demo_entity_set[demo["sample_id"]]]
        demo_output = f"[Example-{indx} Output]:\n```json\n" + json.dumps(demo_entity, ensure_ascii=False, indent=None, sort_keys=False) +"\n```"
        few_shot_content.append({"type": "text", "text": demo_input + demo_output})
        # few_shot_content.append({"type": "text", "text": f"[Example-{indx} Output]:\n"})
        indx +=1
    return few_shot_content

def Refinement_generation(prediction, dataset_name, img_path,  client, output_path, icl_flag, demo_set, demo_entity_set, demo_img_path):
    img_id = prediction[0]+".jpg"
    if len(prediction[2])>0: # 小模型有输出的实体
        pred_json = {"text": prediction[1]["text"], "entity":[{"name": values["name"], "type": values["label"]}for values in prediction[2]]}
        original_text = "\nText and Entities: " + json.dumps(pred_json, ensure_ascii=False, indent=None, sort_keys=False) + "\n"
        try:
            # img = encode_image(img_path + img_id)
            img, scale = resize_image_high_resolution(image_path=img_path + img_id)
            system_prompt = get_system_prompt(dataset_name)
            if icl_flag == True:
                input_text =  original_text
                few_shot_content = get_few_shot_prompt(demo_set, prediction[0], demo_entity_set, demo_img_path)
                raw_content = [
                    {"type": "text", "text": "- Carefully think about the examples above, and give your final Output for **[Original Input]**:\n[Original Input]\nImage:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "text", "text": input_text},
                ]
                input_content = few_shot_content + raw_content
            else:
                input_text =  original_text
                input_content = [
                    {"type": "text", "text": "\n[Original Input]\nImage:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "text", "text": input_text},
                ]
            chat_result = chat(input_content, system_prompt, client)
            Final_result = json_parsing(chat_result)
        except Exception as e:
            print(f"读取图片错误: {e}")
            Final_result = [{"entity": values["name"], "entity_type": values["label"], "bounding_box": []} for values in prediction[2]]

        # print(f"chat_result: {chat_result}")
        # print(f"\nFinal_result: {Final_result}")

        if Final_result:
            for entity in Final_result:
                try:
                    if isinstance(entity["bounding_box"], str):
                        entity["bounding_box"] = str_to_list(entity["bounding_box"])
                    if entity["entity_type"] not in ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]:
                        print("输出错误实体类型: " + entity["entity_type"])
                        entity["entity_type"] = "other"
                except Exception as e:
                    print(f"处理实体时发生错误: {e}")
    else:
        Final_result = None
        scale = 1

    # save_result
    Save_result(Final_result, img_id, scale, output_path)

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


def Save_result(final_result, img_id, scale = None, output_path = None):
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
            if final_result is not None:
                for entity in final_result:
                    if len(entity["bounding_box"])>0:
                        entity["bounding_box"] = resolve_original_image_bbox(entity["bounding_box"], scale)
                        pre_entities.append({"name": entity["entity"], "label": entity["entity_type"], "bnd": {"xmin": entity["bounding_box"][0], "ymin": entity["bounding_box"][1], "xmax": entity["bounding_box"][2], "ymax": entity["bounding_box"][3]}})
                    else:
                        pre_entities.append({"name": entity["entity"], "label": entity["entity_type"], "bnd": None})
            else:
                print(f"image_id 为 {img_id} 时无产生文本实体结果, 跳过视觉绑定")

            # 追加新数据
            # data.append({"image_id": img_id, "pre_entities": pre_entities})
            data[img_id.split(".")[0]] = pre_entities

            # 将更新后的数据写回文件
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"写入 JSON 文件时出错: {e}")

def call_LLM(img_path, input_path, pred_path, icl_demo_path, icl_annotation_path, icl_image_path,output_path, dataset_name, icl_flag = False, max_threads=1, model_name="gemma-3-4b-it", name="base"):

    output_path = output_path + f"/{dataset_name}/" + f"{os.path.splitext(output_path)[1]}/{dataset_name}_{model_name}_{name}_result.json"
    
    Local_predictions = read_json(input_path)
    # refine_pred = read_json(pred_path)
    refine_pred = convert_conll_test_to_json(pred_path, input_path)
    demo_set = read_json(icl_demo_path)
    demo_entity_set = read_json(icl_annotation_path)
    
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
    remaining_predictions = [(key, value, refine_pred[key]) for key, value in Local_predictions.items() if key not in processed_img_ids]

    # Update the model name
    api_config.MODEL_NAME = model_name
    print(f"LLM for visual grounding start, using model: {api_config.MODEL_NAME}")
    print(f"LLM for visual grounding start, using dataset: {dataset_name}")
    client = OpenAI(base_url=api_config.API_BASE, api_key=api_config.API_KEY)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(Refinement_generation, prediction, dataset_name, img_path, client, output_path, icl_flag, demo_set, demo_entity_set, icl_image_path) for prediction in remaining_predictions]
        for future in tqdm(futures, desc="LLM4MNER...", total=len(futures)):
            future.result()

    print("LLM for visual grounding completed!")

if __name__ == '__main__':
    call_LLM(
        img_path = "../data/evaluation_test_set/Evaluation_image/",
        input_path = "../data/evaluation_test_set/Evaluation_text.json",
        pred_path = "/data1/tangjielong/ccks2025/AdaSeq/experiments/CCKS_test_0810/250810133218.223278/test_entity.json",
        icl_demo_path = "/data1/tangjielong/ccks2025/VL_object_detection/utils/output/similarity_weighted_top3.json",
        icl_annotation_path = "/data1/tangjielong/ccks2025/data/sample_entity.json",
        icl_image_path = "/data1/tangjielong/ccks2025/data/sample_image/",
        output_path ="../output",
        dataset_name = "CCKS_test_set",
        model_name = "qwen2.5-vl-72b-instruct",
        name = "3-shot_VisualMapping_20250811",
        icl_flag = True #是否启用in context learning
    )
    
