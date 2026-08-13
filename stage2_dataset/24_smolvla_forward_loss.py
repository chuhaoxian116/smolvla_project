from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

MODEL_ID = "lerobot/smolvla_base"

BATCH_SIZE = 4
CHUNK_SIZE = 10
DEVICE = "cuda"

# Stage 1 已经加载过 smolvla_base。
# 这里优先只使用本机 Hugging Face Cache，避免重复联网。
LOCAL_FILES_ONLY = True


def main():
    print("========== 1. Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

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

    print("\n========== 2. Build SmolVLA Config ==========")

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        device=DEVICE,
        load_vlm_weights=False,
    )

    print(f"Input features  : {list(cfg.input_features.keys())}")
    print(f"Output features : {list(cfg.output_features.keys())}")
    print(f"Chunk size      : {cfg.chunk_size}")
    print(f"Action dim      : {cfg.action_feature.shape[0]}")
    print(f"Device          : {cfg.device}")

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index in cfg.action_delta_indices
        ]
    }

    print("\n========== 3. Dataset + DataLoader ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    raw_batch = next(iter(dataloader))

    print(f"state  : {tuple(raw_batch['observation.state'].shape)}")
    print(f"image  : {tuple(raw_batch['observation.images.camera'].shape)}")
    print(f"action : {tuple(raw_batch['action'].shape)}")
    print(f"pad    : {tuple(raw_batch['action_is_pad'].shape)}")
    print(f"task   : {raw_batch['task'][0]!r}")

    for camera_key in metadata.camera_keys:
        if (
            camera_key in raw_batch
            and isinstance(raw_batch[camera_key], torch.Tensor)
            and raw_batch[camera_key].dtype == torch.uint8
        ):
            raw_batch[camera_key] = (
                raw_batch[camera_key].to(torch.float32) / 255.0
            )

    print("\n========== 4. PreProcessor ==========")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=metadata.stats,
    )

    processed_batch = preprocessor(raw_batch)

    print(
        "state  :",
        tuple(processed_batch["observation.state"].shape),
        processed_batch["observation.state"].device,
    )
    print(
        "image  :",
        tuple(processed_batch["observation.images.camera"].shape),
        processed_batch["observation.images.camera"].device,
    )
    print(
        "action :",
        tuple(processed_batch["action"].shape),
        processed_batch["action"].device,
    )
    print(
        "tokens :",
        tuple(processed_batch["observation.language.tokens"].shape),
        processed_batch["observation.language.tokens"].device,
    )

    print("\n========== 5. Load smolvla_base ==========")

    policy = SmolVLAPolicy.from_pretrained(
        MODEL_ID,
        config=cfg,
        local_files_only=LOCAL_FILES_ONLY,
        strict=False,
    )

    policy.train()

    print(f"Policy device : {next(policy.parameters()).device}")
    print(f"Policy mode   : {'train' if policy.training else 'eval'}")

    print("\n========== 6. SmolVLA Forward ==========")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            loss, loss_dict = policy.forward(processed_batch)

    print(f"loss       : {float(loss):.6f}")
    print(f"loss dtype : {loss.dtype}")
    print(f"loss device: {loss.device}")

    print("\nLoss details:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\nCUDA peak allocated: {peak_gb:.3f} GiB")

    print("\n========== 7. PASS ==========")
    print("自己的 LeRobotDataset")
    print("        ↓")
    print("Action Chunk")
    print("        ↓")
    print("DataLoader Batch")
    print("        ↓")
    print("SmolVLA PreProcessor")
    print("        ↓")
    print("smolvla_base")
    print("        ↓")
    print("Training Forward")
    print("        ↓")
    print("Loss")
    print()
    print("本 Demo 没有执行 loss.backward()，模型参数没有被训练更新。")


if __name__ == "__main__":
    main()