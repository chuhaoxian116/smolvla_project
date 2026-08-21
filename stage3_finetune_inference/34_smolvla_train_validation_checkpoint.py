import json
import shutil
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

CHECKPOINT_DIR = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/34_train_ep01"
)

TRAIN_EPISODES = [0, 1]
VALIDATION_EPISODES = [2]

BATCH_SIZE = 4
CHUNK_SIZE = 10
NUM_EPOCHS = 3

DEVICE = "cuda"
LEARNING_RATE = 1e-4

LOCAL_FILES_ONLY = True
RESET_CHECKPOINT = True


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Dataset Metadata
    # =========================================================================
    print("========== 1. Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    print(f"Train Episodes      : {TRAIN_EPISODES}")
    print(f"Validation Episodes : {VALIDATION_EPISODES}")

    policy_features = dataset_to_policy_features(
        metadata.features
    )

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

    # =========================================================================
    # 2. SmolVLA Config
    # =========================================================================
    print("\n========== 2. SmolVLA Config ==========")

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        device=DEVICE,
        load_vlm_weights=False,
    )

    print(f"Chunk size : {cfg.chunk_size}")
    print(f"Action dim : {cfg.action_feature.shape[0]}")

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index in cfg.action_delta_indices
        ]
    }

    # =========================================================================
    # 3. Full Dataset
    # =========================================================================
    print("\n========== 3. Full Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    print(f"Full Dataset : {len(dataset)} samples")

    # =========================================================================
    # 4. Train / Validation Split
    # =========================================================================
    print("\n========== 4. Train / Validation Split ==========")

    train_indices = []
    validation_indices = []

    for episode_index in TRAIN_EPISODES:
        episode = metadata.episodes[episode_index]

        start = int(
            episode["dataset_from_index"]
        )
        end = int(
            episode["dataset_to_index"]
        )

        train_indices.extend(
            range(start, end)
        )

        print(
            f"Train Episode {episode_index} : "
            f"[{start}, {end})"
        )

    for episode_index in VALIDATION_EPISODES:
        episode = metadata.episodes[episode_index]

        start = int(
            episode["dataset_from_index"]
        )
        end = int(
            episode["dataset_to_index"]
        )

        validation_indices.extend(
            range(start, end)
        )

        print(
            f"Validation Episode {episode_index} : "
            f"[{start}, {end})"
        )

    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices,
    )

    validation_dataset = torch.utils.data.Subset(
        dataset,
        validation_indices,
    )

    print()
    print(f"Train samples      : {len(train_dataset)}")
    print(f"Validation samples : {len(validation_dataset)}")

    # =========================================================================
    # 5. Train DataLoader
    # =========================================================================
    print("\n========== 5. Train DataLoader ==========")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Batches / epoch : {len(train_loader)}")
    print(f"Epochs          : {NUM_EPOCHS}")

    print(
        f"Total steps     : "
        f"{len(train_loader) * NUM_EPOCHS}"
    )

    # =========================================================================
    # 6. Processor
    # =========================================================================
    print("\n========== 6. Processor ==========")

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=cfg,
            dataset_stats=metadata.stats,
        )
    )

    print("PreProcessor  : PASS")
    print("PostProcessor : PASS")

    # =========================================================================
    # 7. Load Base Model
    # =========================================================================
    print("\n========== 7. Load smolvla_base ==========")

    policy = SmolVLAPolicy.from_pretrained(
        MODEL_ID,
        config=cfg,
        local_files_only=LOCAL_FILES_ONLY,
        strict=False,
    )

    policy.train()

    print(
        f"Device : "
        f"{next(policy.parameters()).device}"
    )

    print(
        f"Mode   : "
        f"{'train' if policy.training else 'eval'}"
    )

    trainable_params = [
        param
        for param in policy.parameters()
        if param.requires_grad
    ]

    print(
        f"Trainable parameters : "
        f"{sum(p.numel() for p in trainable_params):,}"
    )

    # =========================================================================
    # 8. Optimizer
    # =========================================================================
    print("\n========== 8. Optimizer ==========")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("Optimizer     : AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    # =========================================================================
    # 9. Training
    # =========================================================================
    print("\n========== 9. Training ==========")

    global_step = 0
    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for raw_batch in train_loader:

            for camera_key in metadata.camera_keys:
                if (
                    camera_key in raw_batch
                    and isinstance(
                        raw_batch[camera_key],
                        torch.Tensor,
                    )
                    and raw_batch[camera_key].dtype
                    == torch.uint8
                ):
                    raw_batch[camera_key] = (
                        raw_batch[camera_key]
                        .to(torch.float32)
                        / 255.0
                    )

            batch = preprocessor(
                raw_batch
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                loss, _ = policy.forward(
                    batch
                )

            loss.backward()
            optimizer.step()

            loss_value = (
                loss.detach().item()
            )

            epoch_loss_sum += loss_value
            epoch_steps += 1
            global_step += 1

        average_loss = (
            epoch_loss_sum
            / epoch_steps
        )

        epoch_losses.append(
            average_loss
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Average Loss = "
            f"{average_loss:.6f}"
        )

    # =========================================================================
    # 10. Save Checkpoint
    # =========================================================================
    print("\n========== 10. Save Checkpoint ==========")

    if (
        CHECKPOINT_DIR.exists()
        and RESET_CHECKPOINT
    ):
        shutil.rmtree(
            CHECKPOINT_DIR
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(
        CHECKPOINT_DIR
    )

    preprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    postprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    training_info = {
        "base_model": MODEL_ID,
        "dataset": REPO_ID,
        "train_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "epochs": NUM_EPOCHS,
        "training_steps": global_step,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "chunk_size": CHUNK_SIZE,
        "epoch_losses": epoch_losses,
    }

    with open(
        CHECKPOINT_DIR / "training_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            training_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Checkpoint : {CHECKPOINT_DIR}")

    # =========================================================================
    # 11. Result
    # =========================================================================
    print("\n========== 11. Result ==========")

    print(
        f"Train Episodes      : "
        f"{TRAIN_EPISODES}"
    )

    print(
        f"Validation Episodes : "
        f"{VALIDATION_EPISODES}"
    )

    print(
        f"Training Steps      : "
        f"{global_step}"
    )

    print(
        f"Final Train Loss    : "
        f"{epoch_losses[-1]:.6f}"
    )

    print()
    print("Episode 0 / 1 participated in training : YES")
    print("Episode 2 participated in training     : NO")

    print("\n========== Main Flow ==========")

    print(
        """
Dataset
   │
   ├── Episode 0 ─┐
   │              ├── Train Dataset
   ├── Episode 1 ─┘       │
   │                      ▼
   │                  DataLoader
   │                      ↓
   │                 PreProcessor
   │                      ↓
   │                  SmolVLA
   │                      ↓
   │           Forward → Loss
   │                      ↓
   │          Backward → Optimizer
   │                      ↓
   │                 Checkpoint
   │
   └── Episode 2 ───── Validation
                         │
                         └── 本 Demo 不使用
"""
    )

    print("========== DONE ==========")

    print()
    print("34 只负责生成 Train/Validation Checkpoint。")
    print()
    print(
        "下一步 35：分别用 Episode 0/1 和 "
        "Episode 2 做 Prediction vs Expert。"
    )


if __name__ == "__main__":
    main()