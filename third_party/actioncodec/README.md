# ActionCodec: What Makes for Good Action Tokenizers

[![Paper](https://img.shields.io/badge/arXiv-2602.15397-b31b1b.svg)](https://arxiv.org/abs/2602.15397)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ActionCodec--Base-yellow)](https://huggingface.co/ZibinDong/ActionCodec-Base)

ActionCodec is a neural codec for encoding and decoding robot action sequences, enabling vision-language-action models to achieve better performance through learned action tokenization.

## Installation

### Step 1: Clone and Setup Conda Environment

```bash
# Clone the repository
git clone https://github.com/ZibinDong/actioncodec.git
cd actioncodec

# Create and activate conda environment
conda env create -f environment.yaml
conda activate actioncodec
```

### Step 2: Install LIBERO

Install the LIBERO benchmark following the official instructions:

```bash
# Visit https://github.com/Lifelong-Robot-Learning/LIBERO for detailed instructions
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

## Main Components

### ActionCodec

A neural codec for encoding/decoding robot action sequences using a Perceiver-based encoder-decoder architecture with Vector Quantization (VQ/RVQ).

```python
from transformers import AutoModel

# Load the base model
model = AutoModel.from_pretrained("ZibinDong/ActionCodec-Base", trust_remote_code=True)

# Or the RVQ fine-tuned variant
model = AutoModel.from_pretrained("ZibinDong/ActionCodec-Base-RVQft", trust_remote_code=True)
```

### SmolVLA Variants

SmolVLA combines a Vision-Language Model (VLM) with an Action Expert for action prediction. Four architecture variants are available:

| Variant | Description |
|---------|-------------|
| **ar** | Auto-Regressive - standard next-token prediction |
| **pd** | Parallel Decoding - one-shot generation |
| **ki** | Knowledge Isolation - continuous actions via flow matching |
| **bar** | Blockwise Autoregressive - block-by-block generation |

**Available Pretrained Models:**
- `ZibinDong/SmolVLM2-2.2B-ActionCodec-PD-LIBERO`
- `ZibinDong/SmolVLM2-2.2B-ActionCodec-AR-LIBERO`
- `ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO`

See [smolvla/README.md](smolvla/README.md) for detailed usage.

### VisionLanguageActionProcessor

A unified processor combining VL processors with action tokenization. Supports five modes:

| Mode | Description |
|------|-------------|
| `discrete` | Expands vocabulary with action tokens |
| `discrete_bar` | Discrete with block boundary markers |
| `mapped` | Maps to existing vocabulary tokens |
| `numeric` | Raw integer sequences |
| `identity` | No tokenization (continuous values) |

See [utils/README.md](utils/README.md) for detailed usage.

## Training and Evaluation

### Training

Use `scripts/train_vla.py` with a config file to train different model variants:

```bash
# Parallel Decoding
python scripts/train_vla.py --config config/train/pd.yaml

# Auto-Regressive
python scripts/train_vla.py --config config/train/ar.yaml

# Blockwise Auto-Regressive
python scripts/train_vla.py --config config/train/bar.yaml

# Knowledge Isolation
python scripts/train_vla.py --config config/train/ki.yaml
```

### Evaluation

Use `scripts/eval_libero.py` to evaluate on LIBERO benchmark:

```bash
# Evaluate a single task
python scripts/eval_libero.py \
    --task_id 0 \
    --ckpt_dir ZibinDong/SmolVLM2-2.2B-ActionCodec-PD-LIBERO \
    --task_suite goal \
    --cfg_path config/eval/pd.yaml

# Or use the shell script for full benchmark evaluation
bash scripts/eval_libero_pd.sh --ckpt_dir ZibinDong/SmolVLM2-2.2B-ActionCodec-PD-LIBERO
```

## Project Structure

```
actioncodec/
├── actioncodec/          # Core codec implementation
├── smolvla/              # VLA model variants (AR, PD, KI, BAR)
├── utils/                # Utilities including VLA tokenizer
├── scripts/              # Training and evaluation scripts
│   ├── train_vla.py      # Unified training script
│   ├── eval_libero.py    # LIBERO evaluation script
│   ├── eval_libero_pd.sh # PD model full benchmark
│   ├── eval_libero_ar.sh # AR model full benchmark
│   └── eval_libero_bar.sh# BAR model full benchmark
├── config/               # Configuration files
│   ├── train/            # Training configs
│   └── eval/             # Evaluation configs
├── environment.yaml      # Conda environment specification
├── requirements.txt      # Python dependencies
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dong2026actioncodecmakesgoodaction,
      title={ActionCodec: What Makes for Good Action Tokenizers},
      author={Zibin Dong and Yicheng Liu and Shiduo Zhang and Baijun Ye and Yifu Yuan and Fei Ni and Jingjing Gong and Xipeng Qiu and Hang Zhao and Yinchuan Li and Jianye Hao},
      year={2026},
      eprint={2602.15397},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.15397},
}
```

## License

This project is released under the MIT License.
