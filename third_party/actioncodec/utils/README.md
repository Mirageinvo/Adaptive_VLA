# VisionLanguageActionProcessor

A unified processor that combines vision-language (VL) processors with action tokenization for robot learning applications.

## Overview

`VisionLanguageActionProcessor` wraps any Hugging Face vision-language processor and adds action tokenization capabilities. It supports five different modes for encoding and decoding robot actions, making it suitable for various model architectures and training approaches.

## Quick Start

```python
from transformers import AutoProcessor
from utils.vla_tokenizer import VisionLanguageActionProcessor
import numpy as np

# Load the action processor and VL processor
action_processor = AutoProcessor.from_pretrained(
    "your-action-processor-path",
    trust_remote_code=True
)
vl_processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)

# Create the unified processor
processor = VisionLanguageActionProcessor(
    action_processor=action_processor,
    vl_processor=vl_processor,
    mode="discrete"  # or "mapped", "numeric", "discrete_bar", "identity"
)

# Process text with actions
action = np.random.randn(20, 7)  # 20 timesteps, 7 action dimensions
text = f"{{'action': {action.tolist()}}}"

result = processor(text=[text], return_tensors="pt", padding=True)
print(result["input_ids"].shape)

# Decode back to actions
decoded = processor.batch_decode(result["input_ids"], decode_actions=True)
print(decoded[0])
```

## Installation

Dependencies:
- `transformers>=4.40.0`
- `numpy`
- `termcolor`
- `huggingface_hub`

```bash
pip install transformers numpy termcolor huggingface_hub
```

## Modes Explained

### 1. `discrete`

Expands the vocabulary with special action tokens (`<|action_0|>`, `<|action_1|>`, etc.) and encodes actions as sequences of these tokens.

**Input format:**
```
{'action': [[0.123, 0.456, ...], [0.789, ...], ...]}
```

**Output format:**
```
{'action': '<|action_42|><|action_17|><|action_89|>...'}
```

**Use case:** When you can modify the model's embedding layer to add new tokens.

**Important:** After loading, resize model embeddings:
```python
model.resize_token_embeddings(len(processor.tokenizer))
```

### 2. `discrete_bar`

Same as `discrete`, but wraps action sequences with block boundary markers.

**Output format:**
```
{'action': '<|blk_bos|><|action_42|><|action_17|>...<|blk_eos|>'}
```

**Use case:** For block-wise autoregressive models that need explicit sequence boundaries.

### 3. `mapped`

Maps actions to existing vocabulary tokens without expanding the vocabulary.

**Output format:**
```
{'action': 'some existing tokens from vocabulary...'}
```

**Use case:** When you cannot modify the embedding layer. Uses tokens from a specified range in the existing vocabulary.

**Configuration:**
```python
processor = VisionLanguageActionProcessor(
    action_processor=action_processor,
    vl_processor=vl_processor,
    mode="mapped",
    vocab_shift=100  # Reserve 100 tokens from the end
)
```

### 4. `numeric`

Encodes actions as raw integer sequences without vocabulary changes.

**Output format:**
```
{'action': [42, 17, 89, 23, ...]}
```

**Use case:** Simple baseline or when you want to process token IDs directly.

### 5. `identity`

No tokenization. Only formats numbers to 2 decimal places for consistency.

**Input format:**
```
{'action': [[0.123456, 0.456789, ...], ...]}
```

**Output format:**
```
{'action': [[0.12, 0.46, ...], ...]}
```

**Use case:** For models that process continuous values directly (e.g., through separate action heads).

## API Reference

### Constructor

```python
VisionLanguageActionProcessor(
    action_processor: Any,      # Action tokenizer with encode/decode methods
    vl_processor: ProcessorMixin,  # Hugging Face VL processor
    mode: str = "numeric",      # One of: discrete, discrete_bar, mapped, numeric, identity
    vocab_shift: int = 0,       # For mapped mode: offset from vocabulary end
)
```

### Key Methods

#### `__call__(*args, **kwargs)`

Process text inputs, converting action dictionaries to tokenized representations.

```python
result = processor(
    text=["Action: {'action': [[0.1, 0.2], [0.3, 0.4]]}"],
    return_tensors="pt",
    padding=True,
    action_processor_kwargs={"embodiment_id": 0}  # Optional
)
```

#### `decode(token_ids, decode_actions=True, precision=8, actions_only=False)`

Decode token IDs back to text with optional action decoding.

```python
# Decode to text with actions
text = processor.decode(input_ids, decode_actions=True, precision=8)

# Get only the action values
actions = processor.decode(input_ids, decode_actions=True, actions_only=True)
```

#### `batch_decode(token_ids, ...)`

Batch version of decode for multiple sequences.

#### `save_pretrained(save_directory)`

Save the processor to a directory.

```python
processor.save_pretrained("./my_processor")
```

Creates:
```
./my_processor/
├── vl_processor/
│   ├── tokenizer.json
│   ├── preprocessor_config.json
│   └── ...
└── action_processor/
    ├── config.json
    └── ...
```

#### `from_pretrained(pretrained_model_name_or_path, **kwargs)`

Load a saved processor.

```python
# From local directory
processor = VisionLanguageActionProcessor.from_pretrained("./my_processor")

# From Hugging Face Hub
processor = VisionLanguageActionProcessor.from_pretrained(
    "username/my-vla-processor",
    token="your-hf-token"  # Optional
)
```

#### `change_mode(new_mode, **kwargs)`

Switch between modes at runtime.

```python
processor.change_mode("mapped", vocab_shift=100)
```

## Integration Guide

### With PyTorch Models

```python
import torch
from transformers import AutoModelForCausalLM

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("your-model")
processor = VisionLanguageActionProcessor(..., mode="discrete")

# Resize embeddings for discrete mode
model.resize_token_embeddings(len(processor.tokenizer))

# Training loop
for batch in dataloader:
    inputs = processor(
        text=batch["text"],
        images=batch["images"],
        return_tensors="pt"
    )
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
```

### With Custom Action Processors

Your action processor should implement:

```python
class MyActionProcessor:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, actions: np.ndarray, **kwargs) -> list:
        """Convert continuous actions to token IDs."""
        # actions: (batch, horizon, action_dim)
        # return: list of token ID lists
        pass

    def decode(self, tokens: list, **kwargs) -> np.ndarray:
        """Convert token IDs back to continuous actions."""
        # tokens: list of token ID lists
        # return: (batch, horizon, action_dim)
        pass

    def save_pretrained(self, path):
        pass

    @classmethod
    def from_pretrained(cls, path):
        pass
```

## Examples

### Basic Encoding/Decoding

```python
import numpy as np

# Create sample action
action = np.random.randn(16, 7)  # 16 timesteps, 7 dimensions
text = f"Robot action: {{'action': {action.tolist()}}}"

# Encode
encoded = processor(text=[text], return_tensors="pt")
input_ids = encoded["input_ids"]

# Decode
decoded = processor.batch_decode(input_ids, decode_actions=True, precision=4)
print(decoded[0])
```

### Chat Template Integration

```python
messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What action should the robot take?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{{'action': {action.tolist()}}}"}
            ]
        }
    ]
]

# Apply chat template
text = processor.apply_chat_template(messages, add_generation_prompt=True)

# Process
inputs = processor(text=text, images=images, return_tensors="pt")
```

### Saving and Loading

```python
# Save
processor.save_pretrained("./checkpoints/my_processor")

# Load
loaded = VisionLanguageActionProcessor.from_pretrained(
    "./checkpoints/my_processor",
    action_processor_kwargs={"trust_remote_code": True}
)
```

## Running Tests

```bash
# Run pytest tests
pytest tests/test_from_pretrained.py -v

# Run with custom checkpoint
TEST_CKPT_DIR=/path/to/checkpoint pytest tests/test_from_pretrained.py -v

# Run the __main__ test script
python utils/vla_tokenizer.py

# With custom paths
ACTION_PROCESSOR_PATH=/path/to/action VL_PROCESSOR_PATH=/path/to/vl python utils/vla_tokenizer.py

# Test specific modes
TEST_MODES=discrete,numeric python utils/vla_tokenizer.py
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

See the main project license.
