

# ReFineG 

This repository provides instructions for reproducing our results in the [CCKS 2025 GMNER Shared Task](https://www.osredm.com/competition/zstp2025/fingerpost). Our system achieved a leaderboard score of **F1: 0.6461, Precision: 0.6261, Recall: 0.6673**, ranking **2nd place** overall.

---

### 0. Environment Requirements

```bash
cuda: 12.2
Recommended GPU resources: 4 × NVIDIA A100 SXM4 / 80GB
```

Since the supervised model training (AdaSeq sequence labeling framework we used) and vLLM framework may have dependency conflicts, we strongly recommend setting them up in **two separate conda environments**. This ensures reproducibility and does not affect the usage of the project.

**Env 1: AdaSeq environment**

```bash
conda create -n adaseq python=3.8
conda activate adaseq
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r ./Local_NER/requirements.txt
```

**Env 2: vLLM environment (open a new terminal)**

```bash
conda create -n vllm python=3.11.11 
conda activate vllm
pip install -r requirements.txt
```

---

### 1. Download Required Dataset

Please obtain official authorization and comply with the dataset license agreement before downloading. Find the dataset in [CCKS 2025 GMNER Shared Task](https://www.osredm.com/competition/zstp2025/fingerpost). Once permission is granted, please place the corresponding files as the following structure in the data file::

```bash
|----data
    |----sample_entity.json # 500 demos multimodal entities.
    |----sample_text.json # 500 demos raw text.
    |----sample_image # 500 demos raw image here.
    |----evaluation_test_set # test set data
    |    |----Evaluation_image # test set raw image here.
    |    |----Evaluation_text.json # test set raw text here.
    |----CCKS_NER_Aug # all LLM-synthetic data here

```

### 2. Download Required Model Resources

* **Download local model checkpoints** (optional)

```bash
conda activate vllm
python download_checkpoints.py
```

* **Download retrieval models**: `all-MiniLM-L6-v2`, `clip-ViT-L-14`

```bash
python retrieval_model_download.py
```

* **Download Qwen2.5-VL-72B model weights**

> The full model requires **≥288GB GPU memory** (recommended: 4 × A100 80GB).
> We use the full 72B model in this project. Quantized versions can be used but may affect accuracy.
> If your resources are limited, consider using the official API ([documentation](https://help.aliyun.com/zh/model-studio/use-qwen-by-calling-api?spm=a2c4g.11186623.0.0.3cf57d9db0YlOa#15ef0a40798a3)), which provides **1M free tokens** (approx. **2M tokens** required for one run).

```bash
python download_qwen2.5vl-72b.py
```

All pretrained models will be stored in `./models`.

---

### 3. Install vLLM (Skip if using the official API)

After setting up environment variables, install **vLLM** following its official instruction [here](https://github.com/vllm-project/vllm).

---

### 4. Quick Reproduction

* **Step 0: Activate AdaSeq environment**

```bash
conda activate adaseq
```

* **Data Augmentation & Preprocessing**
  We augment the initial **500 labeled samples** using an LLM to generate **15k military-domain NER samples**.
  The augmented dataset is stored in `data/CCKS_NER_Aug`.
  Data is formatted into CoNLL, split into **train (3/4)** and **dev (1/4)**.

```bash
sh preprocessing.sh
```

---

* **(Optional) Step 1: Train Local NER Model (xlm-roberta-large-CRF)**

1. Update dataset paths in `Local_NER/examples/CCKS2025/ccks2025_500.yaml`:

```yaml
dataset:
  data_file:
    train: Your_absolute_path/data/CCKS_NER_Aug/CCKS_Aug_train.conll
    valid: Your_absolute_path/data/CCKS_NER_Aug/CCKS_Aug_dev.conll
    test: Your_absolute_path/data/CCKS_NER_Aug/CCKS_test.conll
  data_type: conll
```

2. Run training

```bash
sh train_local_ner.sh
```

Output will be saved to: `experiments/CCKS2025/XXX`

---

* **Step 2: Local NER Model Inference**

1. Update dataset paths in `checkpoints/tjl928/xlm_roberta_large_best_0810/config.yaml`
   (or update `inference_local_ner.sh` if using your own checkpoints).

2. Run inference

```bash
sh inference_local_ner.sh
```

Results will be saved to:
`checkpoints/tjl928/xlm_roberta_large_best_0810/pred.txt`

---

* **Step 3: Adaptive Multimodal Examples Selection Algorithm for In-Context Learning**
  Retrieve top-k demonstrations from the sample set for multimodal LLM inference (default k=3).

```bash
sh cal_sim_top-k_ICL.sh
```

---

* **Step 4: Multimodal LLM Refinement**
  Use **Qwen2.5-VL-72B-Instruct** to refine Local NER outputs and perform **3-shot entity grounding**.

1. Configure API settings in `LLM_baseline/api_config.py`

   Example (local vLLM deployment):

   ```python
   API_TYPE = "open_ai"
   API_BASE = "http://<your_ip>:10091/v1"
   API_KEY = "random"
   MODEL_NAME = "Qwen2.5-VL-72B-Instruct"
   ```

2. Run refinement

```bash
sh inference_MLLM_refinement.sh
```

Final results will be saved in `output/CCKS_2025/xxx`

---

## License & Disclaimer

All code is released for research purposes only and intended for validation by the CCKS 2025 GMNER Shared Task. Make sure to follow the license requirements provided by the CCKS 2025 GMNER organizers.

## Acknowledgements

The datasets we used are from  [CCKS 2025 GMNER Shared Task](https://www.osredm.com/competition/zstp2025/fingerpost). Some codes are based on the open-sourced codes [AdaSeq](https://github.com/modelscope/AdaSeq/tree/master) and [vLLM](https://github.com/vllm-project/vllm). Thanks for their great works!

## Citation
