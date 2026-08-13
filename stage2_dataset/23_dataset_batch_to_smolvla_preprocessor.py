from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.feature_utils import dataset_to_policy_features


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

BATCH_SIZE = 4
CHUNK_SIZE = 10
DEVICE = "cuda"


def print_tensor(name: str, value):
    if isinstance(value, torch.Tensor):
        print(
            f"{name:<40} "
            f"shape={tuple(value.shape)!s:<18} "
            f"dtype={str(value.dtype):<15} "
            f"device={value.device}"
        )
    else:
        print(f"{name:<40} type={type(value).__name__}")


def main():
    print("========== 1. Load Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    # Dataset Feature -> PolicyFeature
    #
    # 这里只保留模型真正关心的：
    # VISUAL / STATE / ACTION。
    policy_features = dataset_to_policy_features(metadata.features)

    output_features = {
        key: feature
        for key, feature in policy_features.items()
        if feature.type is FeatureType.ACTION
    }

    input_features = {
        key: feature
        for key, feature in policy_features.items()
        if key not in output_features
    }

    print("\n========== 2. Dataset -> SmolVLA Features ==========")

    print("Input features:")
    for key, feature in input_features.items():
        print(f"  {key}: type={feature.type}, shape={feature.shape}")

    print("Output features:")
    for key, feature in output_features.items():
        print(f"  {key}: type={feature.type}, shape={feature.shape}")

    # 当前 Demo 使用 10-step Action Chunk。
    # 后面真正按 smolvla_base 训练时，可以再切回 chunk_size=50。
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        device=DEVICE,
    )

    print("\n========== 3. SmolVLA Config ==========")
    print(f"device          : {cfg.device}")
    print(f"chunk_size      : {cfg.chunk_size}")
    print(f"n_action_steps  : {cfg.n_action_steps}")
    print(f"action indices  : {cfg.action_delta_indices}")

    # SmolVLA action_delta_indices = [0, 1, ... chunk_size-1]
    # 根据 Dataset FPS 转成秒。
    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index in cfg.action_delta_indices
        ]
    }

    print("\n========== 4. Build Training Dataset ==========")
    print(f"delta_timestamps[action] = {delta_timestamps['action']}")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    # 整个 Dataset 都交给 DataLoader。
    # shuffle=True：训练时 Sample 可以来自不同 Episode；
    # 但每个 Sample 内的 Action Chunk 仍受 Episode 边界保护。
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    raw_batch = next(iter(dataloader))

    print("\n========== 5. Raw DataLoader Batch ==========")

    for key, value in raw_batch.items():
        print_tensor(key, value)

    print("\nRaw training fields:")
    print_tensor(
        "observation.state",
        raw_batch["observation.state"],
    )
    print_tensor(
        "observation.images.camera",
        raw_batch["observation.images.camera"],
    )
    print_tensor(
        "action",
        raw_batch["action"],
    )

    if "action_is_pad" in raw_batch:
        print_tensor(
            "action_is_pad",
            raw_batch["action_is_pad"],
        )

    if "task" in raw_batch:
        print(f"task[0]                                 = {raw_batch['task'][0]!r}")

    # LeRobot 官方训练脚本会在进入 processor 前，
    # 把 uint8 Camera 转成 [0,1] float32。
    for camera_key in metadata.camera_keys:
        if (
            camera_key in raw_batch
            and isinstance(raw_batch[camera_key], torch.Tensor)
            and raw_batch[camera_key].dtype == torch.uint8
        ):
            raw_batch[camera_key] = (
                raw_batch[camera_key].to(dtype=torch.float32) / 255.0
            )

    print("\n========== 6. Create SmolVLA PreProcessor ==========")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=metadata.stats,
    )

    print(preprocessor)

    # 关键一步：
    #
    # DataLoader Batch
    #       ↓
    # SmolVLA PreProcessor
    #       ↓
    # 模型真正能吃的 Tensor Batch
    processed_batch = preprocessor(raw_batch)

    print("\n========== 7. Processed Batch ==========")

    for key, value in processed_batch.items():
        print_tensor(key, value)

    print("\nImportant processed fields:")

    for key in [
        "observation.state",
        "observation.images.camera",
        "action",
        "action_is_pad",
        "observation.language.tokens",
        "observation.language.attention_mask",
    ]:
        if key in processed_batch:
            print_tensor(key, processed_batch[key])

    print("\n========== 8. Model Handoff ==========")
    print("现在 processed_batch 已经是 SmolVLA.forward(...) 的输入格式。")
    print()
    print("完整链路：")
    print("LeRobotDataset on disk")
    print("        ↓")
    print("delta_timestamps")
    print("        ↓")
    print(f"Action Chunk [{CHUNK_SIZE}, 6]")
    print("        ↓")
    print("DataLoader")
    print("        ↓")
    print(f"Batch Action [{BATCH_SIZE}, {CHUNK_SIZE}, 6]")
    print("        ↓")
    print("SmolVLA PreProcessor")
    print("  - Task -> language tokens")
    print("  - State / Action -> normalization")
    print("  - Tensor -> CUDA")
    print("        ↓")
    print("processed_batch")
    print("        ↓")
    print("下一步：SmolVLA.forward(processed_batch) -> Loss")


if __name__ == "__main__":
    main()