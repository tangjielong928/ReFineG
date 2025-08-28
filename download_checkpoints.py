from modelscope import snapshot_download

# 下载训练的local NER模型
model_dir = snapshot_download('tjl928/xlm_roberta_large_best_0810', cache_dir='./checkpoints')