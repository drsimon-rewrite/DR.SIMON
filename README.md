#  DR.SIMON

**Domain-wise Rewrite for Segment-Informed Medical Oversight Network**

**[Project Page](https://drsimon-rewrite.github.io/)**

---
##  Overview
**DR.SIMON** addresses the long-standing challenge of aligning medical terminology in queries with visual content for temporal grounding in medical videos.

###  Pipeline

<img src="https://github.com/user-attachments/assets/f075a4a8-221c-4f0f-9749-b6499f56cbb8" alt="DR.SIMON Pipeline" width="100%"/>

---

##  Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/drsimon-rewrite/Dr.Simon.git
cd Dr.Simon

# Install dependencies
pip install -r requirements.txt

```

---

##  Dataset Preparation

### 1. Data Setup

Follow the [ReVisionLLM instruction](https://github.com/Tanveer81/ReVisionLLM/tree/main/revisionllm/data) for video feature extraction:

```bash
data/
├── medical/
│   ├── feature/      
│   │   ├── train/
│   │   ├── test/
│   │   └── val/
│   └── rewrited_query/  
│       ├── medvqa_train_grouped_rewrite_from_all_summarized.json
│       └── medvqa_test_grouped_rewrite_from_all_summarized.json
```


### 2.  Model Checkpoints

Download pretrained models following [VTimeLLM instruction](https://github.com/huangb23/VTimeLLM):

```bash
checkpoints/
├── clip/
│   └── ViT-L-14.pt
├── vtimellm-vicuna-v1-5-7b-stage1/
│   └── mm_projector.bin
└── vtimellm-vicuna-v1-5-7b-stage3/
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

---

## Usage

### Quick Start

```bash
python main.py --config config/default.json
```

### Custom Configuration

```bash
python main.py \
  --data_path data/medical/rewrited_query/medvqa_test_grouped_rewrite_from_all_summarized.json \
  --feat_folder data/medical/feature/test \
  --clip_path checkpoints/clip/ViT-L-14.pt \
  --pretrain_mm_mlp_adapter checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage3_path checkpoints/vtimellm-vicuna-v1-5-7b-stage3 \
  --output_path results/predictions.json
```
