"""
Shared utilities for VLA training and evaluation.
"""

import argparse
import importlib
import random
from collections import defaultdict
from functools import partial
from typing import List, Literal

import gym
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from libero.libero import benchmark
from libero2gym.vector_env import AsyncVectorEnv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ColorJitter, Compose, RandomCrop, Resize

from utils.vla_tokenizer import VisionLanguageActionProcessor

# =============================================================================
# Configuration Utilities
# =============================================================================


def get_cfg(cfg_path=None):
    """Load and merge configuration from file and CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode: train 10 steps, no logging, no checkpoints")

    if cfg_path is None:
        parser.add_argument("--config", required=True, help="Path to config file")
        args, remaining_args = parser.parse_known_args()
        base_config = OmegaConf.load(args.config)
        OmegaConf.resolve(base_config)
        base_config.config = args.config
    else:
        args, remaining_args = parser.parse_known_args()
        base_config = OmegaConf.load(cfg_path)
        OmegaConf.resolve(base_config)
        base_config.config = cfg_path
    cli_config = OmegaConf.from_cli(remaining_args)
    merged_config = OmegaConf.merge(base_config, cli_config)
    merged_config.test_mode = args.test
    return merged_config


def get_obj_from_str(string, reload=False):
    """Dynamically import a class from a module path string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def dict_apply(func, d):
    """Apply a function recursively to all non-dict/non-list values in a nested structure."""
    if isinstance(d, dict):
        return {k: dict_apply(func, v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_apply(func, v) for v in d]
    else:
        return func(d)


# =============================================================================
# Random Seed
# =============================================================================


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Prompt Template
# =============================================================================


def prompt_template(
    state: np.ndarray,
    action: np.ndarray | None = None,
    task: str = "",
    mode: Literal["numeric", "discrete", "mapped"] = "numeric",
    action_vocab_size: int = 2048,
    action_token_len: int | None = None,
) -> List[dict]:
    """Create a prompt template for the VLA model."""
    user_content = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {
                "type": "text",
                "text": f"**State**: {[round(s, 3) for s in state.tolist()]}, **Task**: {task}.",
            },
        ],
    }
    if mode == "numeric":
        # reference: https://arxiv.org/pdf/2510.13054v1
        if action_token_len is None:
            action_token_len = "uncertain length"
        template = [
            {
                "role": "system",
                "content": (
                    f"Analyze the input image and predict robot actions. Output a single sequence of {action_token_len} integers (0-{action_vocab_size - 1} each)."
                    "Provide comma-separated numbers as a single-line Python dictionary string containing only the key 'action'. "
                    "Output Example (Format ONLY): {str:[int,int,...,int]}. Nothing else."
                ),
            },
        ]
    elif mode == "discrete":
        template = [
            {
                "role": "system",
                "content": "Analyze the input image and predict robot actions.",
            },
        ]
    elif mode == "mapped":
        template = [
            {
                "role": "system",
                "content": "Analyze the input image and predict robot actions.",
            },
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    template.append(user_content)
    if action is not None:
        template.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "{'action':"
                        + f"{[[round(a, 5) for a in step] for step in action.tolist()]}"
                        + "}",
                    }
                ],
            }
        )
    return template


# =============================================================================
# Normalization Constants
# =============================================================================

# Constants for state normalization
STATE_Q99 = np.array(
    [
        0.13556506,
        0.33566484,
        1.27066591,
        3.27734607,
        2.4061097,
        0.59776972,
        0.04031316,
        -0.00177811,
    ]
)

STATE_Q01 = np.array(
    [
        -0.39912487,
        -0.26883513,
        0.03826696,
        1.50895805,
        -2.71979114,
        -1.08050857,
        0.00174237,
        -0.04002561,
    ]
)

# Constants for action normalization
ACTION_Q99 = np.array(
    [0.9375, 0.9107142686843872, 0.9375, 0.20357142388820648, 0.26357144117355347, 0.375, 1.0]
)

ACTION_Q01 = np.array(
    [-0.8785714507103, -0.875892877579, -0.9375, -0.15107143, -0.20678571, -0.27964285, -1.0]
)

MAX_ACTION_Q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))


# =============================================================================
# Dataset Class
# =============================================================================


class LiberoAllDataset(torch.utils.data.Dataset):
    """Libero dataset for VLA training."""

    def __init__(self, action_vocab_size: int, action_token_len: int | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(
            repo_id="physical-intelligence/libero",
            delta_timestamps={"actions": [i / 10 for i in range(20)]},
        )
        self.resize = RandomCrop((224, 224))

        self.image_transform = Compose(
            [
                RandomCrop(int(256 * 0.875)),
                Resize((224, 224)),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        )

        self.action_vocab_size = action_vocab_size
        self.action_token_len = action_token_len
        self.state_q99 = STATE_Q99
        self.state_q01 = STATE_Q01

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        action_is_pad = item["actions_is_pad"]
        action = item["actions"]
        state = item["state"]

        action_pad_mask = action_is_pad[:, None].repeat(1, action.shape[1])
        action_pad_mask[:, -1] = False
        action = torch.where(action_pad_mask, torch.zeros_like(action), action)

        action[..., :-1] = action[..., :-1] / MAX_ACTION_Q[:-1]
        action[:, -1] = -action[:, -1]

        state = (state - self.state_q01) / (self.state_q99 - self.state_q01) * 2.0 - 1.0
        action = torch.clamp(action, -1.0, 1.0)

        image1 = self.image_transform((item["image"] * 255).to(torch.uint8))
        image2 = self.image_transform((item["wrist_image"] * 255).to(torch.uint8))
        image = torch.cat([image1, image2], dim=2)

        messages = prompt_template(
            state,
            action,
            item["task"],
            mode="discrete",
            action_vocab_size=self.action_vocab_size,
            action_token_len=self.action_token_len,
        )
        messages[1]["content"] = messages[1]["content"][1:]

        return {"messages": messages, "images": [image], "action": action}


# =============================================================================
# Data Collators
# =============================================================================


class VLADataCollator:
    """Data collator for AR strategy - includes full messages with assistant response."""

    def __init__(self, processor: VisionLanguageActionProcessor):
        self.processor = processor

    def __call__(self, features: List[dict]) -> dict:
        all_messages = [f["messages"] for f in features]
        texts = self.processor.apply_chat_template(all_messages, add_generation_prompt=False)
        images = [f["images"] for f in features]
        batch = self.processor(
            action_processor_kwargs={"embodiment_ids": 0},
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batch["action"] = [f["action"] for f in features]
        return batch


class VLDataCollator:
    """Data collator for BAR/PD/KI strategies - messages without assistant response."""

    def __init__(self, processor: VisionLanguageActionProcessor):
        self.processor = processor

    def __call__(self, features: List[dict]) -> dict:
        all_messages = [f["messages"][:-1] for f in features]
        texts = self.processor.apply_chat_template(all_messages, add_generation_prompt=True)
        images = [f["images"] for f in features]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batch["action"] = [f["action"] for f in features]
        return batch


# =============================================================================
# Dataloader Factory
# =============================================================================


def create_dataloaders(cfg, processor, strategy: str):
    """Create train and validation dataloaders."""
    dataset = LiberoAllDataset(
        action_vocab_size=cfg.MODEL.action_processor.vocab_size,
        action_token_len=cfg.MODEL.action_processor.token_len,
    )
    val_ratio = 16 / len(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, [1 - val_ratio, val_ratio])

    # AR strategy uses VLADataCollator for training (includes assistant response)
    train_collator = VLADataCollator(processor) if strategy == "ar" else VLDataCollator(processor)
    val_collator = VLDataCollator(processor)

    train_dataloader = DataLoader(
        train_set,
        batch_size=cfg.TRAINING.batch_size,
        shuffle=True,
        num_workers=cfg.TRAINING.num_workers,
        collate_fn=train_collator,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=cfg.TRAINING.batch_size,
        shuffle=False,
        num_workers=cfg.TRAINING.num_workers,
        collate_fn=val_collator,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


# =============================================================================
# Evaluation Utilities
# =============================================================================


class ActionEnsembler:
    """Ensembler for averaging action predictions across multiple timesteps."""

    def __init__(self):
        self.action_cache = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: torch.Tensor, start_timestamp: int):
        batch_size, horizon, action_dim = action_chunk.shape

        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[:, i, :])

    def get_action(self, timestamp: int) -> torch.Tensor:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        averaged_action = np.mean(stacked_preds, axis=0)
        return averaged_action


def process_state(state):
    """Process raw state into position, orientation, and gripper format."""
    gripper = state[:, :2]
    pos = state[:, 2:5]
    quat = state[:, 5:9]
    ori = np.stack([T.quat2axisangle(each) for each in quat], 0)
    return np.concatenate([pos, ori, gripper], axis=1)


def get_envs(task_suite, env_kwargs, n_envs):
    """Create vectorized environments for the specified task suite."""
    env_fn = partial(gym.make, id=f"libero-{task_suite}-v0", **env_kwargs)
    dummy_env_fn = partial(gym.make, id=f"libero-{task_suite}-v0", dummy=True, **env_kwargs)
    envs = AsyncVectorEnv(
        env_fns=[env_fn for _ in range(n_envs)],
        dummy_env_fn=dummy_env_fn,
    )
    task_id = env_kwargs["task_id"]
    task = benchmark.get_benchmark_dict()[f"libero_{task_suite}"]().get_task(task_id)
    task_description = task.language
    return envs, task_description
