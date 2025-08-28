from modelscope import snapshot_download

# 下载Qwen2.5-VL-72B-Instruct模型
model_dir = snapshot_download('Qwen/Qwen2.5-VL-72B-Instruct', cache_dir='./models')