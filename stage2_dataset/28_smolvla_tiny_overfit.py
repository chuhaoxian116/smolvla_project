# 28_smolvla_tiny_overfit.py

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

EPISODE_INDEX = 0

BATCH_SIZE = 4
CHUNK_SIZE = 10
NUM_EPOCHS = 20

DEVICE = "cuda"
LEARNING_RATE = 1e-4

LOCAL_FILES_ONLY = True


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Dataset Metadata -> 获取 Dataset 和 Episode 信息
    # =========================================================================
    print("========== 1. Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    episode = metadata.episodes[EPISODE_INDEX]

    episode_start = int(episode["dataset_from_index"])
    episode_end = int(episode["dataset_to_index"])
    episode_length = int(episode["length"])

    print(f"\nUse Episode     : {EPISODE_INDEX}")
    print(f"Dataset range   : [{episode_start}, {episode_end})")
    print(f"Episode samples : {episode_length}")

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
    # 3. Dataset -> 创建完整 Dataset，再只选择 Episode 0
    # =========================================================================
    print("\n========== 3. Tiny Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    episode_indices = list(
        range(episode_start, episode_end)
    )

    tiny_dataset = torch.utils.data.Subset(
        dataset,
        episode_indices,
    )

    print(f"Full Dataset : {len(dataset)} samples")
    print(f"Tiny Dataset : {len(tiny_dataset)} samples")
    print(f"Only Episode : {EPISODE_INDEX}")

    # =========================================================================
    # 4. DataLoader -> Episode 0 的 20 个 Sample 组成 Batch
    # =========================================================================
    print("\n========== 4. DataLoader ==========")

    dataloader = torch.utils.data.DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Batches / epoch : {len(dataloader)}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(
        f"Total steps     : "
        f"{len(dataloader) * NUM_EPOCHS}"
    )

    # =========================================================================
    # 5. PreProcessor -> Dataset Batch 转成 SmolVLA 输入
    # =========================================================================
    print("\n========== 5. PreProcessor ==========")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=metadata.stats,
    )

    # =========================================================================
    # 6. Load Model -> 加载预训练 smolvla_base
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

    trainable_count = sum(
        param.numel()
        for param in trainable_params
    )

    print(f"Trainable parameters : {trainable_count:,}")

    # =========================================================================
    # 7. Optimizer -> 根据 Gradient 更新模型
    # =========================================================================
    print("\n========== 7. Optimizer ==========")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("Optimizer     : AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    # =========================================================================
    # 8. Tiny Overfit Training -> 同一个 Episode 反复训练
    # =========================================================================
    print("\n========== 8. Tiny Overfit Training ==========")

    epoch_losses = []
    global_step = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for raw_batch in dataloader:
            # 主干 1：Dataset Batch -> SmolVLA Input
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

            # 主干 2：清除上一轮 Gradient
            optimizer.zero_grad(set_to_none=True)

            # 主干 3：Forward -> Loss
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                loss, _ = policy.forward(batch)

            # 主干 4：Backward -> Gradient
            loss.backward()

            # 主干 5：Optimizer -> Parameter Update
            optimizer.step()

            loss_value = loss.detach().item()

            epoch_loss_sum += loss_value
            epoch_steps += 1
            global_step += 1

        average_loss = (
            epoch_loss_sum / epoch_steps
        )

        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Average Loss = {average_loss:.6f}"
        )

    # =========================================================================
    # 9. Training Result -> 比较训练前后 Loss 趋势
    # =========================================================================
    print("\n========== 9. Training Result ==========")

    first_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    best_loss = min(epoch_losses)

    print(f"First Epoch Loss : {first_loss:.6f}")
    print(f"Final Epoch Loss : {final_loss:.6f}")
    print(f"Best Epoch Loss  : {best_loss:.6f}")

    loss_change = final_loss - first_loss
    loss_ratio = final_loss / first_loss

    print(f"Loss change      : {loss_change:+.6f}")
    print(f"Final / First    : {loss_ratio:.4f}")

    print(
        f"Loss decreased   : "
        f"{final_loss < first_loss}"
    )

    if torch.cuda.is_available():
        peak_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"CUDA peak allocated: "
            f"{peak_gb:.3f} GiB"
        )

    # =========================================================================
    # 10. 主干总结
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
Episode 0
   ↓
20 Samples
   ↓
DataLoader
   ↓
PreProcessor
   ↓
SmolVLA
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Parameter Update
   ↓
重复学习同一个 Episode
   ↓
观察 Loss 是否总体下降
"""
    )

    print("========== DONE ==========")

    print(f"Episode        : {EPISODE_INDEX}")
    print(f"Samples        : {len(tiny_dataset)}")
    print(f"Epochs         : {NUM_EPOCHS}")
    print(f"Training Steps : {global_step}")

    print()
    print("这个 Demo 的目标不是训练出可用模型。")
    print("目标是验证 SmolVLA 能否反复学习极小 Dataset。")
    print()
    print("当前训练后的模型仍然没有保存到磁盘。")


if __name__ == "__main__":
    main()