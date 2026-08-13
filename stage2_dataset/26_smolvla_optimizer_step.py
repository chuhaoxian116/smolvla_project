# 26_smolvla_optimizer_step.py

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

LEARNING_RATE = 1e-4
LOCAL_FILES_ONLY = True


def main():
    # =========================================================================
    # 1. Dataset Metadata -> 获取 Dataset 的 Feature、FPS 等定义
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
    # 2. SmolVLA Config -> 告诉模型输入、输出和 Action Chunk 定义
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
    # 3. LeRobotDataset -> 从磁盘读取 Camera / State / Task / Expert Action
    # =========================================================================
    print("\n========== 3. Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    # =========================================================================
    # 4. DataLoader -> 一次取多个 Sample，组成训练 Batch
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

    raw_batch = next(iter(dataloader))

    print(f"state  : {tuple(raw_batch['observation.state'].shape)}")
    print(f"image  : {tuple(raw_batch['observation.images.camera'].shape)}")
    print(f"action : {tuple(raw_batch['action'].shape)}")
    print(f"pad    : {tuple(raw_batch['action_is_pad'].shape)}")

    for camera_key in metadata.camera_keys:
        if (
            camera_key in raw_batch
            and isinstance(raw_batch[camera_key], torch.Tensor)
            and raw_batch[camera_key].dtype == torch.uint8
        ):
            raw_batch[camera_key] = (
                raw_batch[camera_key].to(torch.float32) / 255.0
            )

    # =========================================================================
    # 5. PreProcessor -> Dataset Batch 转换成 SmolVLA 可以直接接收的数据
    # =========================================================================
    print("\n========== 5. PreProcessor ==========")

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

    # =========================================================================
    # 6. from_pretrained -> 加载已经训练好的 smolvla_base
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

    # =========================================================================
    # 7. Optimizer -> 后面根据 Gradient 真正修改模型 Parameter
    # =========================================================================
    print("\n========== 7. Optimizer ==========")

    trainable_params = [
        param
        for param in policy.parameters()
        if param.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print("Optimizer     : AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    # 找一个可训练参数，用于观察 optimizer.step() 前后的变化。
    probe_name = None
    probe_param = None

    for name, param in policy.named_parameters():
        if param.requires_grad:
            probe_name = name
            probe_param = param
            break

    if probe_param is None:
        raise RuntimeError("没有找到可训练 Parameter。")

    print(f"Probe parameter: {probe_name}")

    # =========================================================================
    # 8. zero_grad -> 每一步训练开始前先清空旧 Gradient
    # =========================================================================
    optimizer.zero_grad(set_to_none=True)

    # =========================================================================
    # 9. Forward -> 把 Batch 送进 SmolVLA，计算当前 Loss
    # =========================================================================
    print("\n========== 8. Forward ==========")

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(DEVICE == "cuda"),
    ):
        loss, loss_dict = policy.forward(processed_batch)

    print(f"Loss : {loss.detach().item():.6f}")

    # =========================================================================
    # 10. backward -> 根据 Loss 计算所有可训练 Parameter 的 Gradient
    # =========================================================================
    print("\n========== 9. Backward ==========")

    loss.backward()

    grad_count = sum(
        1
        for param in policy.parameters()
        if param.requires_grad and param.grad is not None
    )

    print(f"Parameters with Gradient : {grad_count}")

    if probe_param.grad is not None:
        print(
            f"Probe grad norm : "
            f"{probe_param.grad.detach().float().norm().item():.8f}"
        )

    # 保存 optimizer.step() 之前的 Parameter。
    parameter_before = probe_param.detach().clone()

    # =========================================================================
    # 11. optimizer.step -> 根据 Gradient 真正更新模型 Parameter
    # =========================================================================
    print("\n========== 10. Optimizer Step ==========")

    optimizer.step()

    parameter_after = probe_param.detach().clone()

    # =========================================================================
    # 12. Parameter Check -> 验证模型参数是否真的发生变化
    # =========================================================================
    print("\n========== 11. Parameter Check ==========")

    difference = (
        parameter_after.float()
        - parameter_before.float()
    )

    changed_elements = (difference != 0).sum().item()

    print(f"Parameter       : {probe_name}")
    print(
        f"Changed elements: "
        f"{changed_elements} / {difference.numel()}"
    )
    print(
        f"Max |change|    : "
        f"{difference.abs().max().item():.10e}"
    )

    print(
        f"Parameter changed: "
        f"{changed_elements > 0}"
    )

    print("\n========== Main Training Flow ==========")

    print(
        """
LeRobotDataset
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
backward()
      ↓
Gradient
      ↓
optimizer.step()
      ↓
Parameter Update
"""
    )

    print("========== PASS ==========")
    print("完成 SmolVLA 1 个真实 Training Step。")


if __name__ == "__main__":
    main()