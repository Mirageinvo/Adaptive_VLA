# SmolVLA + ActionCodec: Vision-Language-Action Models

SmolVLA + ActionCodec implements vision-language-action models for robotic manipulation and control. This module provides three architecture variants for action prediction, each with different trade-offs between generation speed and modeling capacity.

## Overview

SmolVLA combines a Vision-Language Model (VLM) with an Action Expert for action prediction. The key innovation is the **shared attention mechanism** that enables cross-modal reasoning between visual/textual features and action representations.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SmolVLA Architecture                            │
│                                                                     │
│  ┌─────────────────┐           ┌─────────────────────┐              │
│  │ Vision Encoder  │           │   Action Expert     │              │
│  │   (SigLIP)      │           │   (Llama-based)     │              │
│  └────────┬────────┘           └──────────┬──────────┘              │
│           │                               │                         │
│  ┌────────▼────────┐                      │                         │
│  │  Language Model │                      │                         │
│  │  (SmolLM2)      │                      │                         │
│  └────────┬────────┘                      │                         │
│           │                               │                         │
│           └───────────┬───────────────────┘                         │
│                       │                                             │
│              ┌────────▼────────┐                                    │
│              │ Shared Attention│                                    │
│              │ (Joint Q/K/V)   │                                    │
│              └────────┬────────┘                                    │
│                       │                                             │
│              ┌────────▼────────┐                                    │
│              │  Action Output  │                                    │
│              └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture Variants

### 1. Parallel Decoding (`pd.py`)

**One-shot generation of all action tokens.**

Parallel decoding generates all action tokens in a single forward pass, making it the fastest option for inference.

#### Key Features

- **Single Forward Pass**: All tokens generated simultaneously
- **Bidirectional Action Attention**: Action tokens can attend to each other
- **Cross-Modal Attention**: Action tokens can see all VLM keys
- **Learnable Action Embeddings**: Fixed learnable tokens as action queries

#### Attention Mask Rules

```
VLM Query:    Causal within VLM, cannot see action keys
Action Query: Bidirectional within action, can see all VLM keys
```

#### Usage

```python
from smolvla.pd import SmolVLAParallelDecoding, SmolVLAParallelDecodingConfig
from transformers import AutoConfig

# Create config from existing VLM
vlm_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
config = SmolVLAParallelDecodingConfig.from_vlm_config(
    vlm_config,
    token_budget=16,        # Number of action tokens
    action_vocab_size=2048, # Action vocabulary size
)

# Initialize model
model = SmolVLAParallelDecoding(config)

# Training
outputs = model(
    pixel_values=images,
    input_ids=input_ids,
    labels=action_labels,  # Shape: (batch, token_budget)
    random_position_offset=True,  # Data augmentation
)
loss = outputs.loss

# Generation
action_tokens = model.generate(
    pixel_values=images,
    input_ids=input_ids,
    position_offset=0,  # Position encoding stride
)
# action_tokens: (batch, token_budget)
```

### 2. Knowledge Isolation (`ki.py`)

**Continuous action prediction with flow matching.**

Knowledge Isolation separates VLM and Action Expert paths, using flow matching for continuous action prediction instead of discrete tokens.

#### Key Features

- **Continuous Actions**: Predicts action vectors (e.g., 7-DOF robot poses)
- **Flow Matching Training**: Continuous-time generative model
- **KV Cache Sharing**: Reuses VLM's cache during inference
- **Fourier Time Encoding**: Encodes flow matching timestamps

#### Flow Matching Process

```
Training:
1. Sample noise x_0 ~ N(0, I)
2. Sample time t ~ Uniform(0, 1)
3. Interpolate: x_t = (1-t) * x_0 + t * x_1
4. Predict velocity: v_t = model(x_t, t)
5. Target velocity: x_1 - x_0
6. Loss: MSE(v_t, x_1 - x_0)

Inference (Euler method):
1. Initialize: x_0 ~ N(0, I)
2. For t = 0, dt, 2*dt, ..., 1:
   - v_t = model(x_t, t)
   - x_{t+dt} = x_t + v_t * dt
3. Return x_1
```

#### Usage

```python
from smolvla.ki import SmolVLAKnowledgeIsolation, SmolVLAKnowledgeIsolationConfig

# Create config
config = SmolVLAKnowledgeIsolationConfig.from_vlm_config(
    vlm_config,
    action_horizon=20,  # Number of action steps
    action_dim=7,       # Dimension per action
)

# Initialize model
model = SmolVLAKnowledgeIsolation(config)

# Training
outputs = model(
    pixel_values=images,
    input_ids=input_ids,
    labels=target_actions,  # Shape: (batch, action_horizon, action_dim)
    timestamps=torch.rand(batch_size),  # Optional
)
loss = outputs.loss

# Generation (sampling)
actions = model.sample(
    pixel_values=images,
    input_ids=input_ids,
    sampling_steps=10,  # Number of integration steps
)
# actions: (batch, action_horizon, action_dim)
```

### 3. Blockwise Autoregressive (`bar.py`)

**Block-by-block generation with bidirectional attention within blocks.**

Blockwise AR generates action tokens in fixed-size blocks. Each block uses bidirectional attention internally, while maintaining causal order across blocks.

#### Key Features

- **Block Structure**: Tokens divided into blocks (e.g., 48 tokens / 3 blocks = 16 tokens/block)
- **Within-Block Bidirectional**: All tokens in a block can see each other
- **Cross-Block Causal**: Block i can see blocks 0..i-1
- **Flexible Sampling**: Supports greedy, top-k, top-p, and temperature sampling

#### Blockwise Attention Pattern

```
Block IDs:  [  BOS  | Block0 | Block1 | Block2 ]
            [ 0 0 0 | 0 0 0  | 1 1 1  | 2 2 2  ]

For query in Block1:
- Can see: VLM, BOS, Block0, Block1 (all)
- Cannot see: Block2 (future block)
```

#### Usage

```python
from smolvla.bar import SmolVLABlockwiseAR, SmolVLABlockwiseARConfig

# Create config
config = SmolVLABlockwiseARConfig.from_vlm_config(
    vlm_config,
    token_budget=48,   # Total action tokens
    num_blocks=3,      # Number of blocks (48/3 = 16 tokens per block)
    action_vocab_size=2048,
)

# Initialize model
model = SmolVLABlockwiseAR(config)

# Training
outputs = model(
    pixel_values=images,
    input_ids=input_ids,
    labels=action_labels,  # Shape: (batch, token_budget)
)
loss = outputs.loss

# Generation (greedy)
action_tokens = model.generate(
    pixel_values=images,
    input_ids=input_ids,
    do_sample=False,
)

# Generation (sampling)
action_tokens = model.generate(
    pixel_values=images,
    input_ids=input_ids,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

## Common Components

### LlamaActionExpert

A Llama-based transformer that serves as the action processor. Key characteristics:

- Configurable hidden_size (can differ from VLM)
- Attention dimensions aligned with VLM for shared attention
- Supports GQA (Grouped Query Attention)
- Uses eager attention implementation for bidirectional attention

### LlamaActionExpertConfig

Configuration class with parameters:

| Parameter | Description |
|-----------|-------------|
| `vocab_size` | Action vocabulary size |
| `hidden_size` | Hidden dimension |
| `intermediate_size` | MLP intermediate dimension |
| `num_hidden_layers` | Number of layers (typically matches VLM) |
| `num_attention_heads` | Number of attention heads (must match VLM) |
| `num_key_value_heads` | KV heads for GQA (must match VLM) |

## Model Comparison

| Feature | Parallel Decoding | Knowledge Isolation | Blockwise AR |
|---------|------------------|---------------------|--------------|
| Output Type | Discrete tokens | Continuous vectors | Discrete tokens |
| Generation | Single pass | Iterative sampling | Block-by-block |
| Attention | Bidirectional | Bidirectional | Blockwise |
| Speed | Fastest | Slowest | Medium |
| Action Space | Discrete | Continuous | Discrete |
| Use Case | VQ-VAE tokens | Robot poses | VQ-VAE tokens |

## Loading and Saving

### From Pretrained VLM

```python
# Load from a pretrained SmolVLM checkpoint
model = SmolVLAParallelDecoding.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    token_budget=16,
    action_vocab_size=2048,
)
```

### From Saved VLA Checkpoint

```python
# VLA checkpoints have the structure:
# checkpoint/
#   config.json
#   vlm/
#   action_expert/
#   action_components.bin

model = SmolVLAParallelDecoding.from_pretrained(
    "path/to/checkpoint",
)
```

### Saving

```python
model.save_pretrained("path/to/save")
# Creates:
#   config.json
#   vlm/
#   action_expert/
#   action_components.bin
```

## Position Encoding Strategies

All variants support different position encoding strategies:

1. **Continuous (default)**: Positions grow sequentially
   ```
   VLM: [0, 1, 2, ..., n-1]
   Action: [n, n+1, n+2, ...]
   ```

2. **Random Stride (training augmentation)**: Random intervals
   ```
   Action: [n+2, n+5, n+7, ...]  # Random steps
   ```

3. **Fixed Stride (generation)**: Configurable spacing
   ```
   position_offset=0: [n, n+1, n+2, ...]
   position_offset=1: [n+1, n+3, n+5, ...]
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

This code is part of the ActionCodec project.
