# 模型下载
from modelscope import snapshot_download

# 下载all-MiniLM-L6-v2模型
model_dir = snapshot_download('sentence-transformers/all-MiniLM-L6-v2', cache_dir='./models')
# 下载clip-ViT-L-14模型
model_dir = snapshot_download('sentence-transformers/clip-ViT-L-14', cache_dir='./models')
# 下载Qwen2.5-VL-72B-Instruct模型
# model_dir = snapshot_download('Qwen/Qwen2.5-VL-72B-Instruct', cache_dir='./models')