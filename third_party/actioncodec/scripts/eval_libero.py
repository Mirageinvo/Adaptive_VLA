import json
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/dzb/github/libero")

import argparse
import logging

import imageio
import libero  # noqa: F401
import libero2gym  # noqa: F401
import numpy as np
import torch
from utils import (
    ACTION_Q01,
    ACTION_Q99,
    STATE_Q01,
    STATE_Q99,
    ActionEnsembler,
    dict_apply,
    get_cfg,
    get_envs,
    process_state,
    prompt_template,
    seed_everything,
)
from torchvision.transforms.v2 import CenterCrop, Compose, Resize
from transformers import AutoModelForImageTextToText

import actioncodec  # noqa: F401
from smolvla.bar import SmolVLABlockwiseAR
from smolvla.pd import SmolVLAParallelDecoding
from utils.vla_tokenizer import VisionLanguageActionProcessor


def evaluate(args):
    """Main evaluation function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    task_id = args.task_id
    ckpt_dir = args.ckpt_dir
    device = args.device
    task_suite = args.task_suite
    img_aug = args.img_aug
    tiled_img = args.tiled_img
    normalization = args.normalization
    horizon = args.horizon
    ensemble = args.ensemble
    pos_offset = args.pos_offset
    waiting_steps = args.waiting_steps
    seed = args.seed
    output_dir = args.output_dir
    n_envs = args.n_envs

    # Validate constraint: k_set * n_envs = 50
    if 50 % n_envs != 0:
        raise ValueError(f"n_envs must divide 50 evenly. Got n_envs={n_envs}")
    k_set = 50 // n_envs

    seed_everything(seed)

    env_kwargs = {"task_id": task_id, "image_size": 224}
    cfg = get_cfg(args.cfg_path)

    logger.info(f"Initializing {n_envs} environments...")
    envs, task_description = get_envs(task_suite, env_kwargs, n_envs)

    # Print task info at the beginning
    task_suite_display = "long" if task_suite == "10" else task_suite
    print(f"Task Suite: {task_suite_display}")
    print(f"Task ID: {task_id}")
    print(f"Task Description: {task_description}")

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    cfg.TRAINING.ckpt_dir = ckpt_dir

    logger.info(f"Loading model from {ckpt_dir}...")
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = cfg.TRAINING.ckpt_dir
    if "bar" in args.cfg_path:
        model = SmolVLABlockwiseAR.from_pretrained(**cfg.MODEL.vlm.kwargs).to(device, dtype).eval()
    elif "pd" in args.cfg_path:
        model = (
            SmolVLAParallelDecoding.from_pretrained(**cfg.MODEL.vlm.kwargs).to(device, dtype).eval()
        )
    else:
        model = (
            AutoModelForImageTextToText.from_pretrained(**cfg.MODEL.vlm.kwargs)
            .to(device, dtype)
            .eval()
        )

    processor = VisionLanguageActionProcessor.from_pretrained(
        cfg.TRAINING.ckpt_dir, trust_remote_code=True, mode="discrete"
    )
    try:
        _ = processor.action_processor(np.random.randn(2, horizon, 7))
    except Exception:
        pass

    max_act_q = np.maximum(np.abs(ACTION_Q99), np.abs(ACTION_Q01))

    if img_aug:
        image_transform = Compose([CenterCrop(int(224 * 0.875)), Resize(224)])
    else:
        image_transform = Compose([Resize(224)])

    # Create output directories
    task_output_dir = os.path.join(output_dir, task_suite_display, str(task_id))
    os.makedirs(task_output_dir, exist_ok=True)

    all_reward = []
    test_results = []
    video_idx = 0

    if ensemble:
        action_ensembler = ActionEnsembler()

    for k in range(k_set):
        logger.info(f"Starting evaluation round {k + 1}/{k_set}...")

        if ensemble:
            action_ensembler.reset()
            current_timestamp = 0

        # Reset all environments, each with a different initial state
        obs = envs.reset(options=[{"init_state_id": i + k * n_envs} for i in range(n_envs)])
        reward = np.zeros(n_envs)

        # Initial waiting steps
        dummy_action = np.array([[0, 0, 0, 0, 0, 0, -1] for _ in range(n_envs)])
        for _ in range(waiting_steps):
            obs, reward_tmp, done, info = envs.step(dummy_action)
            reward = np.clip(reward + reward_tmp, 0, 1)

        # Create video writers for each environment
        writers = []
        for i in range(n_envs):
            video_path = os.path.join(task_output_dir, f"video_{video_idx + i:03d}.mp4")
            writers.append(imageio.get_writer(video_path, fps=30))

        plt_image_history = [[] for _ in range(n_envs)]

        while not np.all(done):
            state = (process_state(obs["state"]) - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0
            messages = list()

            image1 = torch.tensor(obs["agentview_image"][:, :, ::-1].copy()).permute(0, 3, 1, 2)
            image2 = torch.tensor(obs["robot0_eye_in_hand_image"][:, :, ::-1].copy()).permute(
                0, 3, 1, 2
            )
            if img_aug:
                image1 = image_transform(image1)
                image2 = image_transform(image2)
            if tiled_img:
                image = torch.cat([image1, image2], dim=-1)

            for i in range(n_envs):
                messages.append(
                    prompt_template(
                        state[i],
                        None,
                        task_description,
                        mode=cfg.MODEL.vla_processor.kwargs.mode,
                        action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                        action_token_len=cfg.MODEL.action_processor.token_len,
                    )
                )
                if tiled_img:
                    messages[-1][1]["content"] = messages[-1][1]["content"][1:]

            if tiled_img:
                images = [[image[i].numpy()] for i in range(n_envs)]
            else:
                images = [[image1[i].numpy(), image2[i].numpy()] for i in range(n_envs)]

            texts = processor.apply_chat_template(messages, add_generation_prompt=True)

            processed_batch = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                action_processor_kwargs={"embodiment_ids": 0},
            )

            processed_batch = dict_apply(lambda x: x.to(device, dtype), processed_batch)

            with torch.no_grad():
                if isinstance(model, SmolVLABlockwiseAR) or isinstance(
                    model, SmolVLAParallelDecoding
                ):
                    pred_action_tokens = model.generate(
                        **processed_batch,
                        position_offset=pos_offset,
                        do_sample=False,
                        initial_position_shift=1,
                    )
                    _action = processor.action_processor.decode(pred_action_tokens.tolist())[0]
                else:
                    generate_ids = model.generate(**processed_batch, max_new_tokens=1000)
                    _action = processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=False,
                        actions_only=True,
                    )
                    for i in range(len(_action)):
                        if _action[i] is None:
                            _action[i] = np.zeros((20, 7))
                    _action = np.clip(np.array(_action), -1, 1)

            action = np.copy(_action)
            if normalization == "type0":
                action[..., :-1] = action[..., :-1] * max_act_q[..., :-1]
                action[..., -1] = -action[..., -1]
            else:
                action = (action - ACTION_Q01) / (ACTION_Q99 - ACTION_Q01) * 2.0 - 1.0

            if ensemble:
                action_ensembler.add_actions(action, current_timestamp)

            for t in range(horizon):
                if ensemble:
                    action_to_apply = action_ensembler.get_action(current_timestamp)
                    current_timestamp += 1
                else:
                    action_to_apply = action[:, t]

                obs, reward_tmp, done, info = envs.step(action_to_apply)
                reward = np.clip(reward + reward_tmp, 0, 1)

                # Save frames for each environment
                for i in range(n_envs):
                    plt_image_history[i].append(obs["agentview_image"][i].copy())

        # Write videos and record results
        for i in range(n_envs):
            for frame in plt_image_history[i]:
                writers[i].append_data(frame)
            writers[i].close()

            # Record test result
            test_results.append({"test_id": video_idx + i, "success": bool(reward[i] >= 1.0)})

        all_reward.append(reward)
        video_idx += n_envs

    # Clean up environments
    try:
        envs.close()
    except Exception:
        pass

    all_reward = np.concatenate(all_reward, axis=0)
    success_count = int(np.sum(all_reward >= 1.0))
    total_count = len(all_reward)
    success_rate = success_count / total_count

    # Save results to JSON
    results = {
        "task_suite": task_suite_display,
        "task_id": task_id,
        "task_description": task_description,
        "total_tests": total_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "test_results": test_results,
    }

    json_path = os.path.join(task_output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Success rate: {success_rate * 100:.1f}%")
    print(f"Success Rate: {success_count}/{total_count} ({success_rate * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="path/to/checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task_suite", type=str, default="goal")
    parser.add_argument("--img_aug", type=bool, default=True)
    parser.add_argument("--tiled_img", type=bool, default=True)
    parser.add_argument("--normalization", type=str, default="type0")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--ensemble", type=bool, default=True)
    parser.add_argument("--pos_offset", type=int, default=4)
    parser.add_argument("--waiting_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./evaluation/")
    parser.add_argument("--cfg_path", type=str, default="config/eval/pd.yaml")
    parser.add_argument("--n_envs", type=int, default=10, help="Number of parallel environments")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for model (bfloat16, float16, float32)",
    )
    args = parser.parse_args()
    evaluate(args)
