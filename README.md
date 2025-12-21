# MedVLThinker: Simple Baselines for Multimodal Medical Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2508.02669)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-orange)](https://ucsc-vlaa.github.io/MedVLThinker/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue)](https://huggingface.co/collections/UCSC-VLAA/medvlthinker-688f52224fb7ff7d965d581d)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**MedVLThinker** is an open-source recipe for building *reasoning-centric* medical vision-language models.
It bundles everything you need: cleaned text-only and/or image-text datasets, a difficulty-aware data-curation pipeline, and two turnkey training modes (SFT or RLVR), to reproduce or extend our **state-of-the-art baselines on six public medical-VQA benchmarks**. By scaling the same recipe from 3 B to 32 B parameters, we show that an open 32 B model can **match GPT-4o** on accuracy while remaining fully transparent and reproducible.

## 📰 News

**2025-12-21** — We’re excited to introduce **MedVLSynther** (https://github.com/UCSC-VLAA/MedVLSynther), a rubric-guided **generator–verifier** framework that synthesizes high-quality **multiple-choice medical VQA** items directly from open biomedical literature by grounding on figures, captions, and in-text references. Built from PubMed Central, it releases **MedSynVQA** (13,087 audited questions over 14,803 images across 13 imaging modalities and 28 anatomical regions), and provides a complementary data-generation pipeline you can pair with MedVLThinker’s open training recipes (SFT/RLVR) for reasoning-centric medical vision-language modeling. 


## 🔥 Highlights

* **Fully open stack** – code, filtered datasets, checkpoints, and evaluation scripts are all released under permissive licenses.
* **Simple but strong** – two recipes: supervised fine-tuning (SFT) or reinforcement learning with verifiable rewards (RLVR) on curated data.
* **RLVR > SFT** – RLVR consistently beats SFT across model sizes; on a 7 B backbone it lifts average accuracy from 53.5 % to **54.9 %**.
* **State-of-the-art 7 B** – our 7 B RLVR model tops all previous open medical LMMs on six benchmarks.
* **GPT-4o-level 32 B** – scaling the same recipe to 32 B parameters reaches GPT-4o parity (63 % avg.) while staying open.
* **Data you can trust** – medium-difficulty questions are auto-filtered via pass-count analysis; noisy items are dropped before training.

## 📋 Table of Contents

- [Installation](#-Installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Models and Results](#-models-and-results)
- [Citation](#-citation)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8 or later
- Docker (recommended)

### Option 1: Docker Setup (Recommended)

```bash
git clone git@github.com:UCSC-VLAA/MedVLThinker.git
cd MedVLThinker

# Clone VERL for reinforcement learning
git clone https://github.com/volcengine/verl.git third_party/verl
cd third_party/verl
git checkout 54b2677
cd ../..

# Start Docker container
docker pull whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3

docker run -itd \
--runtime=nvidia \
--gpus all \
--net=host \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--cap-add=SYS_ADMIN \
-v $(pwd):$(pwd) \
-v $HOME:$HOME \
-w $(pwd) \
-e HF_HOME=$(pwd)/cache/ \
-u $(id -u):$(id -g) \
-e HOME=$HOME \
-e USER=$USER \
--memory 900g \
--name MedVLThinker \
whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 \
bash

docker exec -it MedVLThinker bash
pip3 install -e third_party/verl[vllm]
```

### Option 2: Local Installation

```bash
git clone git@github.com:UCSC-VLAA/MedVLThinker.git
cd MedVLThinker

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets qwen-vl-utils
pip install vllm python-dotenv wandb click tqdm matplotlib pandas
```

### Environment Configuration

Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=MedVLThinker
WANDB_MODE=online
WANDB_ENTITY=your_entity

HF_TOKEN=your_huggingface_token
HF_HOME=cache/
```

## 🎯 Quick Start

### Demo

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load the model
model_name="UCSC-VLAA/MedVLThinker-3B-RL_m23k"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Example usage
messages_1 = [
    {
        "role": "system",
        "content": "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/slake_closed.jpg",
            },
            {"type": "text", "text": "Which side of lung is abnormal in this image, left or right?"},
        ],
    }
]

messages_2 = [
    {
        "role": "system",
        "content": "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/MedXpertQA-MM.jpg",
            },
            {"type": "text", "text": "You are shown images of the right and left distal common carotid arteries, respectively. What is the MOST likely diagnosis?"},
        ],
    }
]

# Preparation for inference
messages = messages_2

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.6, top_p=0.95, do_sample=True)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```




## 📊 Datasets

### Available Datasets

Our project provides several curated datasets for medical vision-language understanding and training:

| Dataset | Modality | Description | Download | 
|---------|-----|-------------|----------|
| **MedVLThinker-m23k-tokenized** | Text-only | Tokenized version of the [m23k](https://github.com/UCSC-VLAA/m1) dataset | [🤗 HF](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-m23k-tokenized) |
| **MedVLThinker-pmc_vqa-gpt_4o_reasoning-tokenized** | Image-Text | Tokenized PMC-VQA dataset with GPT-4o generated reasoning chains | [🤗 HF](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-pmc_vqa-gpt_4o_reasoning-tokenized) |
| **MedVLThinker-pmc_vqa** | Image-Text |Processed PMC-VQA dataset for medical visual question answering with RLVR | [🤗 HF](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-pmc_vqa) |
| **MedVLThinker-Eval** | Image-Text|  Comprehensive evaluation dataset for medical VQA benchmarks | [🤗 HF](https://huggingface.co/datasets/UCSC-VLAA/MedVLThinker-Eval) |

### Dataset Usage

```python
from datasets import load_dataset

# Load evaluation dataset
eval_dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval")

# Load training dataset with reasoning
train_dataset = load_dataset("UCSC-VLAA/MedVLThinker-pmc_vqa-gpt_4o_reasoning-tokenized")

# Load PMC-VQA dataset
pmc_dataset = load_dataset("UCSC-VLAA/MedVLThinker-pmc_vqa")

# Load Medical23k tokenized dataset
m23k_dataset = load_dataset("UCSC-VLAA/MedVLThinker-m23k-tokenized")
```

<details><summary>Dataset details and preparation of your own</summary>

### Supported Datasets

Our framework supports evaluation on the following medical VQA datasets:

- **PMC-VQA**: PubMed Central Visual Question Answering
- **PathVQA**: Pathology Visual Question Answering  
- **SLAKE**: Bilingual medical VQA dataset
- **VQA-RAD**: Radiology Visual Question Answering
- **MMMU Medical**: Medical subsets from MMMU benchmark
- **MedXpertQA**: Expert-level medical questions

### Data Format

All datasets follow a unified format:

```python
{
    "images": [PIL.Image],           # List of images
    "question": str,                 # Question text
    "options": Dict[str, str],       # Multiple choice options
    "answer_label": str,             # Correct answer label (A, B, C, D)
    "answer": str,                   # Full answer text
    "reasoning": str,                # Chain-of-thought reasoning (optional)
    "dataset_name": str,             # Source dataset name
    "dataset_index": int             # Unique sample identifier
}
```

### Prepare Evaluation Data

```bash
# Download and prepare evaluation datasets
python data_process/prepare_vlm_eval_data_v2.py

# The processed dataset will be available at: UCSC-VLAA/MedVLThinker-Eval
```

### Prepare Training Data with Pass Rate

```bash
# Estimate pass rates for curriculum learning
bash eval/estimate_pass_rate.sh

# Order training data from easy to hard
python data_process/train_dataset/order_easy_to_hard.py \
    --dataset_name UCSC-VLAA/MedVLThinker-pmc_vqa \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/results.jsonl \
    --save_to_disk_path data/local/pmc_vqa_easy_to_hard
```

</details>

## 🏋️ Training

### Supervised Fine-tuning (SFT)

```bash
train/sft/train_commands.sh
# text sft: train/sft/sft_local.sh
# image-text sft: train/sft/sft_vlm_local.sh

# convert SFT weights
train/sft/convert_weights.py
```

Training logs: https://wandb.ai/xk-huang/med-vlrm/workspace?nw=nwuserxkhuang

### Reinforcement Learning (GRPO)

```bash
bash train/*.sh

# Convert VERL checkpoints for inference
python third_party/verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/medvl_grpo/step_1000/actor \
    --target_dir outputs/converted/medvl-thinker-7b
```

## 🔍 Evaluation

### Comprehensive Evaluation

```bash
# Run evaluation on all medical benchmarks
bash eval/eval_commands.sh

bash eval/eval_baselines.sh

# Analyze results
python analysis/analyze_pass_rate_v2.py
python analysis/result_viewer.py
```

<details><summary>Detailed commands</summary>

### Download Pre-trained Models

```bash
# Download base models
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-3B-Instruct')"
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"
```

### Run Evaluation on Medical Benchmarks

```bash
# Evaluate on medical VQA benchmarks
python eval/run_offline_inference_v2.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dp_size 1 \
    --tp_size 1 \
    --temperature 0.0 \
    --max_tokens 4096 \
    --n 1 \
    --batch_size 32 \
    --output_dir outputs/evaluation
```

### Inference with Trained Models

```bash
# Use our trained medical VLM
python eval/run_offline_inference_v2.py \
    --model UCSC-VLAA/MedVLThinker-7B-RL_m23k \
    --dp_size 1 \
    --tp_size 1 \
    --temperature 0.0 \
    --batch_size 32 \
    --output_dir outputs/medvl-thinker
```

</details>


## 📈 Models and Results

### Available Models

| Model | Size | Training Method | Training Data | Download |
|-------|------|----------------|---------------|----------|
| **SFT Models** |
| MedVLThinker-3B-SFT_m23k | 3B | SFT | m23k | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-SFT_m23k) |
| MedVLThinker-7B-SFT_m23k | 7B | SFT | m23k | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-SFT_m23k) |
| MedVLThinker-3B-SFT_PMC | 3B | SFT | PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-SFT_PMC) |
| MedVLThinker-7B-SFT_PMC | 7B | SFT | PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-SFT_PMC) |
| **RL Models** |
| **MedVLThinker-3B-RL_m23k** | 3B | RL | m23k | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-RL_m23k) |
| **MedVLThinker-7B-RL_m23k** | 7B | RL | m23k | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-RL_m23k) |
| MedVLThinker-3B-RL_PMC | 3B | RL | PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-RL_PMC) |
| MedVLThinker-7B-RL_PMC | 7B | RL | PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-RL_PMC) |
| **MedVLThinker-32B-RL_m23k** | 32B | RL | m23k | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-32B-RL_m23k) |
| **SFT + RL Models** |
| MedVLThinker-3B-SFT_m23k-RL_PMC | 3B | SFT + RL | m23k → PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-SFT_m23k-RL_PMC) |
| MedVLThinker-7B-SFT_m23k-RL_PMC | 7B | SFT + RL | m23k → PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-SFT_m23k-RL_PMC) |
| MedVLThinker-3B-RL_m23k-RL_PMC | 3B | RL + RL | m23k → PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-RL_m23k-RL_PMC) |
| MedVLThinker-7B-RL_m23k-RL_PMC | 7B | RL + RL | m23k → PMC-VQA | [🤗 HF](https://huggingface.co/UCSC-VLAA/MedVLThinker-7B-RL_m23k-RL_PMC) |

### Benchmark Results

3B and 7B with different training recipes.

| Model                   | PMC    | MMMU   | MedX-M | PathVQA | SLAKE  | VQA-Rad | Avg.   |
| :---------------------- | :----: | :----: | :----: | :-----: | :----: | :-----: | :----: |
| Qwen2\.5-VL-3B-Instruct | 44\.77 | 44\.12 | 20\.69 | 61\.96  | 61\.30 | 62\.01  | 49\.14 |
| SFT(m23k)              | 28\.53 | 32\.55 | 16\.00 | 42\.74  | 43\.91 | 33\.09  | 32\.80 |
| SFT(PMC)               | 54\.55 | 47\.84 | 21\.46 | 52\.76  | 65\.79 | 58\.58  | 50\.16 |
| SFT(m23k)+RL(PMC)     | 46\.32 | 44\.31 | 20\.52 | 43\.85  | 58\.49 | 50\.98  | 44\.08 |
| RL(m23k)               | 47\.32 | 52\.16 | 22\.90 | 62\.28  | 63\.38 | 71\.08  | 53\.19 |
| RL(PMC)                | 54\.22 | 48\.43 | 21\.51 | 51\.61  | 75\.56 | 62\.38  | 52\.28 |
| RL(m23k)+RL(PMC)      | 51\.33 | 48\.43 | 22\.60 | 49\.71  | 66\.11 | 60\.17  | 49\.72 |
| Qwen2\.5-VL-7B-Instruct | 49\.30 | 52\.94 | 18\.89 | 65\.39  | 65\.71 | 68\.75  | 53\.50 |
| SFT(m23k)              | 34\.58 | 46\.86 | 16\.40 | 56\.35  | 54\.97 | 53\.80  | 43\.83 |
| SFT(PMC)               | 54\.67 | 49\.80 | 21\.39 | 53\.02  | 67\.71 | 57\.72  | 50\.72 |
| SFT(m23k)+RL(PMC)     | 43\.18 | 47\.84 | 21\.84 | 51\.43  | 60\.34 | 55\.15  | 46\.63 |
| RL(m23k)               | 50\.67 | 56\.86 | 24\.43 | 66\.83  | 65\.79 | 64\.71  | 54\.88 |
| RL(PMC)                | 55\.38 | 55\.29 | 24\.11 | 57\.09  | 66\.59 | 63\.48  | 53\.66 |
| RL(m23k)+RL(PMC)      | 56\.37 | 50\.98 | 25\.80 | 48\.24  | 59\.13 | 58\.09  | 49\.77 |

Comparison with other methods.

| Model                      | PMC    | MMMU   | MedX-M | PathVQA | SLAKE  | VQA-Rad |  Avg.   |
| :------------------------- | :----: | :----: | :----: | :-----: | :----: | :-----: |  :----: |
| General LMM                |        |        |        |         |        |         |         |
| GPT-4o-mini                | 51\.90 | 63\.53 | 28\.55 | 63\.33  | 75\.24 | 66\.91  |  58\.24 |
| GPT-4o                     | 58\.55 | 68\.82 | 35\.95 | 72\.43  | 76\.44 | 70\.22  |  63\.74 |
| Gemme 3 4B                 | 44\.42 | 46\.67 | 21\.89 | 59\.24  | 66\.59 | 56\.86  |  49\.28 |
| Gemme 3 27B                | 52\.05 | 60\.78 | 30\.80 | 65\.70  | 72\.60 | 65\.20  |  57\.86 |
| Qwen2\.5-VL-3B-Instruct    | 44\.77 | 44\.12 | 20\.69 | 61\.96  | 61\.30 | 62\.01  |  49\.14 |
| Qwen2\.5-VL-7B-Instruct    | 49\.30 | 52\.94 | 18\.89 | 65\.39  | 65\.71 | 68\.75  |  53\.50 |
| Qwen2\.5-VL-32B-Instruct   | 53\.28 | 63\.92 | 27\.68 | 67\.98  | 73\.24 | 75\.12  |  60\.20 |
| Medical LMM                |        |        |        |         |        |         |         |
| MedGemma 4B                | 42\.73 | 32\.55 | 8\.17  | 59\.64  | 83\.49 | 78\.55  |  50\.86 |
| MedGemma 27B               | 36\.75 | 35\.88 | 12\.13 | 62\.09  | 77\.40 | 72\.67  |  49\.49 |
| Llava Med v1\.5 Mistral 7B | 34\.28 | 31\.37 | 22\.56 | 56\.52  | 62\.82 | 56\.74  |  44\.05 |
| HuatuoGPT-Vision-7B        | 53\.39 | 50\.59 | 22\.00 | 63\.53  | 75\.00 | 63\.60  |  54\.69 |
| HuatuoGPT-Vision-34B       | 52\.54 | 57\.06 | 21\.80 | 66\.72  | 78\.85 | 74\.26  |  58\.54 |
| MedVLThinker-3B RL(m23k)   | 47\.32 | 52\.16 | 22\.90 | 62\.28  | 63\.38 | 71\.08  |  53\.19 |
| MedVLThinker-7B RL(m23k)   | 50\.67 | 56\.86 | 24\.43 | 66\.83  | 65\.79 | 64\.71  |  54\.88 |
| MedVLThinker-32B RL(m23k)  | 54\.37 | 70\.00 | 34\.60 | 68\.82  | 73\.96 | 76\.96  |  63\.12 |


## 📁 Project Structure

```
MedVLThinker/
├── analysis/           # Result analysis and visualization
├── data_process/       # Data preprocessing and preparation
├── docs/              # Documentation
├── eval/              # Evaluation scripts and benchmarks
├── train/             # Training scripts and configurations
├── third_party/       # External dependencies (VERL)
├── outputs/           # Experiment outputs and results
└── README.md          # This file
```


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [VERL](https://github.com/volcengine/verl) for reinforcement learning framework
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for base vision-language models
- Medical VQA dataset providers

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{huang2025medvlthinker,
  title={Medvlthinker: Simple baselines for multimodal medical reasoning},
  author={Huang, Xiaoke and Wu, Juncheng and Liu, Hui and Tang, Xianfeng and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2508.02669},
  year={2025}
}
@article{m1_2025,
  title={m1: Unleash the potential of test-time scaling for medical reasoning with large language models},
  author={Huang, Xiaoke and Wu, Juncheng and Liu, Hui and Tang, Xianfeng and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2504.00869},
  year={2025}
}
```
