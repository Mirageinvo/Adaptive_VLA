"""
Shared utilities for VLA training and evaluation.
"""

import argparse
import importlib
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.vla_tokenizer import VisionLanguageActionProcessor

# =============================================================================
# Configuration Utilities
# =============================================================================


def get_cfg(cfg_path=None):
    """Load and merge configuration from file and CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode: train 10 steps, no logging, no checkpoints")
    parser.add_argument(
        "--debug-overfit",
        action="store_true",
        help=(
            "Repeat one already-collated real batch for 50 optimizer steps and "
            "enable supervision/gradient/distributed sanity checks."
        ),
    )

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
    merged_config.debug_overfit = args.debug_overfit
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
        from torchvision.transforms.v2 import ColorJitter, Compose, RandomCrop, Resize

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


class CalvinDToDDataset(torch.utils.data.Dataset):
    """Real CALVIN D→D RGB/action windows from the streamed clean cache.

    Expected per-frame keys are exactly the stream extractor contract:
    ``rgb_static``, ``rgb_gripper``, ``rel_actions``, ``robot_obs``, plus
    optional metadata. No depth or tactile array is loaded.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_vocab_size: int,
        action_token_len: int,
        chunk_length: int = 30,
        allow_partial_stream: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.chunk_length = int(chunk_length)
        self.action_vocab_size = int(action_vocab_size)
        self.action_token_len = int(action_token_len)
        self.training_root = self._find_training_root(self.root)
        on_disk = self._available_frame_ids(self.training_root)
        if len(on_disk) < self.chunk_length:
            raise RuntimeError(
                f"need >= {self.chunk_length} on-disk episode frames under "
                f"{self.training_root}; found {len(on_disk)}"
            )
        annotation_path = self._find_language_annotations(self.training_root)
        self.samples: list[tuple[int, str]] = []
        if annotation_path is not None:
            annotations = np.load(annotation_path, allow_pickle=True).item()
            language = annotations.get("language", {})
            info = annotations.get("info", {})
            texts = list(language.get("ann", []))
            bounds = np.asarray(info.get("indx", []), dtype=np.int64)
            if len(texts) == 0 or bounds.ndim != 2 or bounds.shape[1] != 2:
                raise RuntimeError(
                    f"invalid CALVIN auto_lang_ann.npy at {annotation_path}: "
                    f"texts={len(texts)}, bounds_shape={bounds.shape}"
                )
            if len(texts) != len(bounds):
                raise RuntimeError(
                    f"CALVIN language/bounds mismatch: {len(texts)} != {len(bounds)}"
                )
            available = set(on_disk)
            for text, (start, end) in zip(texts, bounds):
                last_start = int(end) - self.chunk_length + 1
                for frame_id in range(int(start), last_start + 1):
                    window = range(frame_id, frame_id + self.chunk_length)
                    if all(fid in available for fid in window):
                        self.samples.append((frame_id, str(text)))
        elif allow_partial_stream:
            # Stream may still be downloading lang_annotations. Build windows
            # only over contiguous on-disk RGB frames so --debug-overfit can run.
            self.samples = self._windows_from_available_frames(on_disk, self.chunk_length)
            print(
                f"[CalvinDToDDataset] auto_lang_ann missing; using "
                f"{len(self.samples)} partial-stream windows from on-disk RGB",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"expected auto_lang_ann.npy under {self.training_root}"
            )
        if not self.samples:
            raise RuntimeError(
                f"no complete {self.chunk_length}-frame windows under {self.training_root}"
            )

    @staticmethod
    def _find_training_root(root: Path) -> Path:
        direct = root / "training"
        if (direct / "ep_start_end_ids.npy").is_file() or any(direct.glob("episode_*.npz")):
            return direct
        candidates = sorted(root.glob("**/training"))
        candidates = [
            path for path in candidates
            if (path / "ep_start_end_ids.npy").is_file() or any(path.glob("episode_*.npz"))
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one CALVIN training tree below {root}, found "
                f"{[str(path) for path in candidates]}"
            )
        return candidates[0]

    @staticmethod
    def _find_language_annotations(training_root: Path) -> Path | None:
        candidates = [
            training_root / "lang_annotations" / "auto_lang_ann.npy",
            training_root / "auto_lang_ann.npy",
        ]
        for path in candidates:
            if path.is_file():
                return path
        found = sorted(training_root.glob("**/auto_lang_ann.npy"))
        if not found:
            return None
        if len(found) != 1:
            raise RuntimeError(
                f"expected one auto_lang_ann.npy below {training_root}, "
                f"found={found}"
            )
        return found[0]

    @staticmethod
    def _available_frame_ids(training_root: Path) -> list[int]:
        ids = []
        for path in training_root.glob("episode_*.npz"):
            try:
                ids.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(ids)

    @staticmethod
    def _windows_from_available_frames(frame_ids: list[int], chunk_length: int) -> list[tuple[int, str]]:
        available = set(frame_ids)
        samples: list[tuple[int, str]] = []
        for frame_id in frame_ids:
            window = range(frame_id, frame_id + chunk_length)
            if all(fid in available for fid in window):
                samples.append((frame_id, "partial_stream_debug_window"))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, frame_id: int) -> dict:
        path = self.training_root / f"episode_{frame_id:07d}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"CALVIN frame missing: {path}. Wait for stream extraction to finish "
                "before running --debug-overfit."
            )
        with np.load(path, allow_pickle=False) as frame:
            keys = set(frame.files)
            forbidden = sorted(
                key for key in keys if key.startswith("depth") or "tactile" in key
            )
            if forbidden:
                raise AssertionError(
                    f"unclean CALVIN frame {path}: forbidden keys={forbidden}"
                )
            required = {"rgb_static", "rgb_gripper", "rel_actions", "robot_obs"}
            missing = required.difference(keys)
            if missing:
                raise AssertionError(f"{path} missing keys={sorted(missing)}")
            return {key: frame[key].copy() for key in required}

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        start, task = self.samples[int(idx)]
        first = self._load_frame(start)
        actions = [first["rel_actions"]]
        for frame_id in range(start + 1, start + self.chunk_length):
            actions.append(self._load_frame(frame_id)["rel_actions"])
        action = torch.from_numpy(np.stack(actions).astype(np.float32, copy=False))
        state = torch.from_numpy(first["robot_obs"].astype(np.float32, copy=False))
        images = [
            Image.fromarray(first["rgb_static"].astype(np.uint8, copy=False)),
            Image.fromarray(first["rgb_gripper"].astype(np.uint8, copy=False)),
        ]
        messages = prompt_template(
            state,
            action,
            task,
            mode="discrete",
            action_vocab_size=self.action_vocab_size,
            action_token_len=self.action_token_len,
        )
        return {"messages": messages, "images": images, "action": action}


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
    dataset_root = None
    if "DATASET" in cfg:
        dataset_root = cfg.DATASET.get("root")
    dataset_root = dataset_root or os.environ.get("CALVIN_DATA_ROOT")
    if dataset_root:
        dataset = CalvinDToDDataset(
            dataset_root,
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len,
            chunk_length=int(cfg.get("DATASET", {}).get("chunk_length", 30)),
        )
        is_calvin = True
    else:
        dataset = LiberoAllDataset(
            action_vocab_size=cfg.MODEL.action_processor.vocab_size,
            action_token_len=cfg.MODEL.action_processor.token_len,
        )
        is_calvin = False
    n = len(dataset)
    val_n = min(16, max(1, n // 10))
    if n <= 2:
        train_set, val_set = dataset, dataset
    else:
        generator = torch.Generator().manual_seed(0)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n - val_n, val_n], generator=generator
        )

    # AR strategy uses VLADataCollator for training (includes assistant response)
    train_collator = VLADataCollator(processor) if strategy == "ar" else VLDataCollator(processor)
    val_collator = VLDataCollator(processor)

    train_dataloader = DataLoader(
        train_set,
        batch_size=cfg.TRAINING.batch_size,
        # FSDP re-executes this script per rank; shuffle=False + shared seed
        # keeps the debug-overfit first batch identical across ranks.
        shuffle=not bool(cfg.get("debug_overfit", False)),
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
    # Explicit provenance gate consumed by --debug-overfit.
    train_dataloader._calvin_real_rgb_cache = is_calvin
    train_dataloader._calvin_dataset_root = str(dataset_root or "")

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
    import robosuite.utils.transform_utils as T

    gripper = state[:, :2]
    pos = state[:, 2:5]
    quat = state[:, 5:9]
    ori = np.stack([T.quat2axisangle(each) for each in quat], 0)
    return np.concatenate([pos, ori, gripper], axis=1)


def get_envs(task_suite, env_kwargs, n_envs):
    """Create vectorized environments for the specified task suite."""
    import gym
    from libero.libero import benchmark
    from libero2gym.vector_env import AsyncVectorEnv

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
