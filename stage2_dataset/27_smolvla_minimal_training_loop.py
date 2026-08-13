# 27_smolvla_minimal_training_loop.py

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
NUM_EPOCHS = 3

DEVICE = "cuda"
LEARNING_RATE = 1e-4

LOCAL_FILES_ONLY = True


def main():
    # =========================================================================
    # 1. Dataset Metadata -> 获取 Dataset 的 Feature / FPS
    # =========================================================================
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
    # 3. Dataset -> 从磁盘读取训练数据
    # =========================================================================
    print("\n========== 3. Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    print(f"Dataset samples : {len(dataset)}")

    # =========================================================================
    # 4. DataLoader -> 多个 Sample 组成一个 Batch
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
    # 7. Optimizer -> 根据 Gradient 更新模型 Parameter
    # =========================================================================
    print("\n========== 7. Optimizer ==========")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("Optimizer     : AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    # 记录一个 Parameter，验证连续训练后模型确实发生变化。
    probe_name = None
    probe_param = None

    for name, param in policy.named_parameters():
        if param.requires_grad:
            probe_name = name
            probe_param = param
            break

    if probe_param is None:
        raise RuntimeError("没有找到可训练 Parameter。")

    parameter_before_training = (
        probe_param.detach().clone().float().cpu()
    )

    # =========================================================================
    # 8. Training Loop -> 重复 Forward / Backward / Optimizer Step
    # =========================================================================
    print("\n========== 8. Training Loop ==========")

    global_step = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        print(
            f"\n---------- Epoch "
            f"{epoch + 1}/{NUM_EPOCHS} ----------"
        )

        for raw_batch in dataloader:
            # =============================================================
            # 主干 1：Dataset Batch -> SmolVLA Input
            # =============================================================

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

            # =============================================================
            # 主干 2：zero_grad -> 清除上一次 Step 的 Gradient
            # =============================================================

            optimizer.zero_grad(set_to_none=True)

            # =============================================================
            # 主干 3：Forward -> 输入 SmolVLA，计算 Loss
            # =============================================================

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                loss, _ = policy.forward(batch)

            # =============================================================
            # 主干 4：Backward -> 根据 Loss 计算 Gradient
            # =============================================================

            loss.backward()

            # =============================================================
            # 主干 5：Optimizer Step -> 真正更新模型 Parameter
            # =============================================================

            optimizer.step()

            loss_value = loss.detach().item()

            epoch_loss_sum += loss_value
            epoch_steps += 1
            global_step += 1

            print(
                f"Step {global_step:03d} | "
                f"Loss = {loss_value:.6f}"
            )

        average_epoch_loss = (
            epoch_loss_sum / epoch_steps
        )

        print(
            f"Epoch {epoch + 1} Average Loss: "
            f"{average_epoch_loss:.6f}"
        )

    # =========================================================================
    # 9. Parameter Check -> 验证多次 optimizer.step() 后模型已经变化
    # =========================================================================
    print("\n========== 9. Parameter Check ==========")

    parameter_after_training = (
        probe_param.detach().clone().float().cpu()
    )

    difference = (
        parameter_after_training
        - parameter_before_training
    )

    changed_elements = (
        difference != 0
    ).sum().item()

    print(f"Probe parameter  : {probe_name}")

    print(
        f"Changed elements : "
        f"{changed_elements} / {difference.numel()}"
    )

    print(
        f"Max |change|     : "
        f"{difference.abs().max().item():.10e}"
    )

    print(
        f"Parameter changed: "
        f"{changed_elements > 0}"
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
    print("\n========== Main Training Flow ==========")

    print(
        """
Dataset
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
zero_grad()
   ↓
backward()
   ↓
Gradient
   ↓
optimizer.step()
   ↓
Parameter Update
   ↓
Next Batch
   ↓
重复训练
"""
    )

    print("========== PASS ==========")

    print(f"Epochs         : {NUM_EPOCHS}")
    print(f"Training Steps : {global_step}")
    print()
    print("SmolVLA 已经执行多次 Parameter Update。")
    print()
    print("注意：当前模型还没有保存到磁盘。")


if __name__ == "__main__":
    main()