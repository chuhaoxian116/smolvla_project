# 29_smolvla_save_checkpoint.py

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
    "checkpoints/29_smolvla_demo"
)

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
    # 1. Dataset Metadata -> 获取 Dataset Feature / FPS
    # =========================================================================
    print("========== 1. Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

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
    # 2. SmolVLA Config -> 定义模型输入 / 输出 / Action Chunk
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

    print(f"Input features  : {list(cfg.input_features.keys())}")
    print(f"Output features : {list(cfg.output_features.keys())}")
    print(f"Chunk size      : {cfg.chunk_size}")
    print(f"Action dim      : {cfg.action_feature.shape[0]}")

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index in cfg.action_delta_indices
        ]
    }

    # =========================================================================
    # 3. Dataset -> 读取全部 Episode
    # =========================================================================
    print("\n========== 3. Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    print(f"Dataset samples : {len(dataset)}")

    # =========================================================================
    # 4. DataLoader -> Sample 组成训练 Batch
    # =========================================================================
    print("\n========== 4. DataLoader ==========")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Batches / epoch  : {len(dataloader)}")
    print(f"Epochs           : {NUM_EPOCHS}")
    print(
        f"Total train steps: "
        f"{len(dataloader) * NUM_EPOCHS}"
    )

    # =========================================================================
    # 5. PreProcessor -> Dataset Batch 转成模型输入
    # =========================================================================
    print("\n========== 5. Pre/Post Processor ==========")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=metadata.stats,
    )

    # =========================================================================
    # 6. Load Model -> 加载 smolvla_base
    # =========================================================================
    print("\n========== 6. Load smolvla_base ==========")

    policy = SmolVLAPolicy.from_pretrained(
        MODEL_ID,
        config=cfg,
        local_files_only=LOCAL_FILES_ONLY,
        strict=False,
    )

    policy.train()

    print(f"Policy device : {next(policy.parameters()).device}")
    print(f"Policy mode   : {'train' if policy.training else 'eval'}")

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
    # 7. Optimizer -> 用 Gradient 更新模型参数
    # =========================================================================
    print("\n========== 7. Optimizer ==========")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("Optimizer     : AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    # =========================================================================
    # 8. Training Loop -> 多次更新模型 Parameter
    # =========================================================================
    print("\n========== 8. Training ==========")

    global_step = 0
    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for raw_batch in dataloader:

            # Dataset Batch -> SmolVLA Input
            for camera_key in metadata.camera_keys:
                if (
                    camera_key in raw_batch
                    and isinstance(
                        raw_batch[camera_key],
                        torch.Tensor,
                    )
                    and raw_batch[camera_key].dtype == torch.uint8
                ):
                    raw_batch[camera_key] = (
                        raw_batch[camera_key].to(torch.float32)
                        / 255.0
                    )

            batch = preprocessor(raw_batch)

            # 清除上一 Step 的 Gradient
            optimizer.zero_grad(set_to_none=True)

            # Forward -> Loss
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                loss, _ = policy.forward(batch)

            # Backward -> Gradient
            loss.backward()

            # Optimizer -> Parameter Update
            optimizer.step()

            loss_value = loss.detach().item()

            epoch_loss_sum += loss_value
            epoch_steps += 1
            global_step += 1

        average_loss = epoch_loss_sum / epoch_steps
        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Average Loss = {average_loss:.6f}"
        )

    # =========================================================================
    # 9. Save Checkpoint -> 保存训练后的 Policy + Processor
    # =========================================================================
    print("\n========== 9. Save Checkpoint ==========")

    if CHECKPOINT_DIR.exists() and RESET_CHECKPOINT:
        shutil.rmtree(CHECKPOINT_DIR)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 保存训练后的 SmolVLA 权重 + Config
    policy.save_pretrained(
        CHECKPOINT_DIR
    )

    # 保存训练时使用的 PreProcessor
    preprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    # 保存推理输出需要的 PostProcessor
    postprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    # 保存简单的训练信息，方便后续检查。
    training_info = {
        "base_model": MODEL_ID,
        "dataset": REPO_ID,
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

    print(f"Checkpoint saved to:")
    print(CHECKPOINT_DIR)

    # =========================================================================
    # 10. Check Files -> 确认 Checkpoint 已经真正写入磁盘
    # =========================================================================
    print("\n========== 10. Checkpoint Files ==========")

    for path in sorted(CHECKPOINT_DIR.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 ** 2)

            print(
                f"{path.relative_to(CHECKPOINT_DIR)} "
                f"({size_mb:.2f} MB)"
            )

    model_file = CHECKPOINT_DIR / "model.safetensors"
    config_file = CHECKPOINT_DIR / "config.json"

    print("\n========== 11. Check ==========")

    print(
        f"model.safetensors : "
        f"{'PASS' if model_file.exists() else 'FAIL'}"
    )

    print(
        f"config.json       : "
        f"{'PASS' if config_file.exists() else 'FAIL'}"
    )

    print(
        f"training_info.json: "
        f"{'PASS' if (CHECKPOINT_DIR / 'training_info.json').exists() else 'FAIL'}"
    )

    # =========================================================================
    # 12. 主干总结
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
LeRobotDataset
      ↓
DataLoader
      ↓
PreProcessor
      ↓
smolvla_base
      ↓
Training Loop
      ↓
optimizer.step()
      ↓
Fine-tuned Parameter
      ↓
policy.save_pretrained()
      ↓
Checkpoint on Disk
"""
    )

    print("========== DONE ==========")

    print(f"Training Steps : {global_step}")
    print(f"Checkpoint     : {CHECKPOINT_DIR}")
    print()
    print("训练后的模型已经保存到磁盘。")
    print("下一步：30_smolvla_reload_checkpoint.py")


if __name__ == "__main__":
    main()