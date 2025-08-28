import json
import os
import re

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_entities(text, entities):
    tags = ['O'] * len(text)
    sorted_entities = sorted(entities, key=lambda x: len(x['name']), reverse=True)
    for ent in sorted_entities:
        ent_name = ent['name']
        ent_type = ent['type']
        start = 0
        while True:
            idx = text.find(ent_name, start)
            if idx == -1:
                break
            if all(t == 'O' for t in tags[idx:idx+len(ent_name)]):
                tags[idx] = f'B-{ent_type}'
                for i in range(1, len(ent_name)):
                    tags[idx+i] = f'I-{ent_type}'
            start = idx + 1
    return tags

def split_text(text):
    # 分词，保留英文单词、数字、标点，且将's等分开
    # 先将 's 拆开
    text = re.sub(r"(\w)('s)\b", r"\1 \2", text)
    # 用正则分割，保留所有分隔符
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

def get_token_tags(text, tags):
    # 将BIO标签按分词对齐
    token_spans = []
    idx = 0
    tokens = []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        start, end = match.start(), match.end()
        tokens.append(text[start:end])
        token_spans.append((start, end))
    token_tags = []
    for (start, end) in token_spans:
        # 取该token区间内出现最多的标签
        tag_count = {}
        for i in range(start, end):
            tag = tags[i]
            tag_count[tag] = tag_count.get(tag, 0) + 1
        # 优先B-，否则I-，否则O
        if not tag_count:
            token_tags.append('O')
        elif any(k.startswith('B-') for k in tag_count):
            b_tags = [k for k in tag_count if k.startswith('B-')]
            token_tags.append(b_tags[0])
        elif any(k.startswith('I-') for k in tag_count):
            i_tags = [k for k in tag_count if k.startswith('I-')]
            token_tags.append(i_tags[0])
        else:
            token_tags.append('O')
    return tokens, token_tags

def convert_NERAug_to_conll(json_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for group in json_data.values():
            for item in group:
                text = item['text']
                entities = item.get('entity', [])
                tags = find_entities(text, entities)
                tokens, token_tags = get_token_tags(text, tags)
                for token, tag in zip(tokens, token_tags):
                    fout.write(f"{token}\t{tag}\n")
                fout.write("\n")

def convert_dev500_to_conll(sample_entity, sample_text, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for key,group in sample_entity.items():
            entities = []
            text = sample_text[key]['text']
            if len(group)>0:
                for item in group:
                    entities.append({"name": item["name"], "type": item["label"]})
            tags = find_entities(text, entities)
            tokens, token_tags = get_token_tags(text, tags)
            for token, tag in zip(tokens, token_tags):
                fout.write(f"{token}\t{tag}\n")
            fout.write("\n")

def convert_test_to_conll(sample_text, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for key, item in sample_text.items():
            text = item['text']
            # 分词，保留英文单词、数字、标点，且将's等分开
            text_mod = re.sub(r"(\w)('s)\b", r"\1 \2", text)
            tokens = re.findall(r"\w+|[^\w\s]", text_mod)
            for token in tokens:
                fout.write(f"{token}\tO\n")
            fout.write("\n")

def convert_conll_test_to_json(pred_path, sample_text_path):
    """
    pred.txt每行格式: token\tO\tPRED_LABEL\n
    sample_text.json: {key: {"text": ...}}
    输出json: {key: [ {"name": 实体片段, "label": 标签, "bnd": {"xmin":0, "ymin":0, "xmax":0, "ymax":0} } ]}
    支持BIOES标注法。实体name直接用原文text切片，避免因分词导致的多余空格。
    """
    with open(sample_text_path, 'r', encoding='utf-8') as f:
        sample_text = json.load(f)
    keys = list(sample_text.keys())
    with open(pred_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    samples = []
    cur = []
    for line in lines:
        if line.strip() == '':
            if cur:
                samples.append(cur)
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        samples.append(cur)
    assert len(samples) == len(keys), f"样本数不一致: pred={len(samples)}, text={len(keys)}"
    result = {}
    for key, sample_lines in zip(keys, samples):
        text = sample_text[key]['text']
        # 重新分词，获得token在原文中的起止位置
        text_mod = re.sub(r"(\w)('s)\b", r"\1 \2", text)
        token_spans = []
        for match in re.finditer(r"\w+|[^\w\s]", text_mod):
            start, end = match.start(), match.end()
            token_spans.append((start, end))
        tokens = [text_mod[s:e] for s, e in token_spans]
        # 读取预测标签
        labels = []
        for line in sample_lines:
            parts = line.split('\t')
            if len(parts) == 3:
                token, _, label = parts
            elif len(parts) == 2:
                token, label = parts
            else:
                continue
            labels.append(label)
        assert len(tokens) == len(labels), f"token数与label数不一致: {len(tokens)} vs {len(labels)}"
        entities = []
        i = 0
        while i < len(tokens):
            label = labels[i]
            if label.startswith('B-'):
                ent_type = label[2:]
                start_span = token_spans[i][0]
                j = i + 1
                while j < len(tokens) and labels[j] == f'I-{ent_type}':
                    j += 1
                if j < len(tokens) and labels[j] == f'E-{ent_type}':
                    end_span = token_spans[j][1]
                    ent_name = text[start_span:end_span]
                    entities.append({
                        "name": ent_name,
                        "label": ent_type,
                        "bnd": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
                    })
                    i = j + 1
                else:
                    # BIO不完整，跳过
                    i = j
            elif label.startswith('S-'):
                ent_type = label[2:]
                start_span, end_span = token_spans[i]
                ent_name = text[start_span:end_span]
                entities.append({
                    "name": ent_name,
                    "label": ent_type,
                    "bnd": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
                })
                i += 1
            else:
                i += 1
        result[key] = entities
    return result

def transfer_train_set():
    input_json = os.path.join(os.path.dirname(__file__), '/data1/tangjielong/ccks2025/data/CCKS_NER_Aug_test/CCKS_NER_Aug_test_qwen-max-latest_20250810_result.json')
    output_conll = os.path.join(os.path.dirname(__file__), '/data1/tangjielong/ccks2025/data/CCKS_NER_Aug_test/CCKS_NER_Aug_test_qwen-max-latest_20250810_result.conll')
    data = load_json(input_json)
    convert_NERAug_to_conll(data, output_conll)

def transfer_dev_set():
    sample_entity = load_json("../evaluation_test_set/new_test_entity_0810.json")
    sample_text = load_json("../evaluation_test_set/Evaluation_text.json")
    convert_dev500_to_conll(sample_entity,sample_text,output_path="./CCKS_dev_0810.conll")

def transfer_test_set():
    sample_text = load_json("../evaluation_test_set/Evaluation_text.json")
    convert_test_to_conll(sample_text,output_path="./CCKS_test.conll")

def revert_test_set():
    result = convert_conll_test_to_json(pred_path="/data1/tangjielong/ccks2025/AdaSeq/experiments/CCKS_test_0810/250810133218.223278/pred.txt",
                               sample_text_path="../evaluation_test_set/Evaluation_text.json")
    
    with open("/data1/tangjielong/ccks2025/AdaSeq/experiments/CCKS_test_0810/250810133218.223278/test_entity.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    revert_test_set()
