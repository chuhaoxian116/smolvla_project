# 30_smolvla_reload_checkpoint.py

import json
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import DataProcessorPipeline


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

CHECKPOINT_DIR = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/29_smolvla_demo"
)

BATCH_SIZE = 4
DEVICE = "cuda"


def main():
    # =========================================================================
    # 1. Checkpoint Files -> 确认保存的模型文件存在
    # =========================================================================
    print("========== 1. Checkpoint Files ==========")

    required_files = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "training_info.json",
    ]

    for filename in required_files:
        path = CHECKPOINT_DIR / filename

        print(
            f"{filename:<30} "
            f"{'PASS' if path.exists() else 'FAIL'}"
        )

        if not path.exists():
            raise FileNotFoundError(path)

    # =========================================================================
    # 2. Training Info -> 查看这个 Checkpoint 是怎么训练出来的
    # =========================================================================
    print("\n========== 2. Training Info ==========")

    with open(
        CHECKPOINT_DIR / "training_info.json",
        "r",
        encoding="utf-8",
    ) as f:
        training_info = json.load(f)

    for key, value in training_info.items():
        print(f"{key:<20}: {value}")

    # =========================================================================
    # 3. Reload Model -> 从本地 Checkpoint 恢复训练后的 SmolVLA
    # =========================================================================
    print("\n========== 3. Reload SmolVLA ==========")

    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_DIR,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    print(f"Policy device : {next(policy.parameters()).device}")
    print(f"Policy mode   : {'train' if policy.training else 'eval'}")
    print(f"Chunk size    : {policy.config.chunk_size}")
    print(f"Action dim    : {policy.config.action_feature.shape[0]}")

    # =========================================================================
    # 4. Reload Processor -> 恢复训练时使用的输入/输出处理器
    # =========================================================================
    print("\n========== 4. Reload Processors ==========")

    preprocessor = DataProcessorPipeline.from_pretrained(
        CHECKPOINT_DIR,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
    )

    postprocessor = DataProcessorPipeline.from_pretrained(
        CHECKPOINT_DIR,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
    )

    print("PreProcessor  : PASS")
    print("PostProcessor : PASS")

    # =========================================================================
    # 5. Parameter Check -> 对比磁盘权重与 Reload 后模型 Parameter
    # =========================================================================
    print("\n========== 5. Parameter Check ==========")

    model_file = CHECKPOINT_DIR / "model.safetensors"

    preferred_probe = (
        "model.vlm_with_expert.lm_expert.layers.0."
        "self_attn.q_proj.weight"
    )

    policy_state = policy.state_dict()

    with safe_open(
        str(model_file),
        framework="pt",
        device="cpu",
    ) as f:
        checkpoint_keys = set(f.keys())

        if (
            preferred_probe in checkpoint_keys
            and preferred_probe in policy_state
        ):
            probe_name = preferred_probe
        else:
            common_keys = [
                key
                for key in f.keys()
                if key in policy_state
            ]

            if not common_keys:
                raise RuntimeError(
                    "Checkpoint 和 Reload Policy 没有共同 Parameter。"
                )

            probe_name = common_keys[0]

        checkpoint_tensor = f.get_tensor(probe_name)

    reloaded_tensor = (
        policy_state[probe_name]
        .detach()
        .cpu()
    )

    exact_equal = torch.equal(
        checkpoint_tensor,
        reloaded_tensor,
    )

    max_diff = (
        checkpoint_tensor.float()
        - reloaded_tensor.float()
    ).abs().max().item()

    print(f"Probe parameter : {probe_name}")
    print(f"Shape           : {tuple(reloaded_tensor.shape)}")
    print(f"Dtype           : {reloaded_tensor.dtype}")
    print(f"Exact equal     : {exact_equal}")
    print(f"Max difference  : {max_diff:.10e}")

    if not exact_equal:
        raise RuntimeError(
            "Reload 后 Parameter 与 model.safetensors 不一致。"
        )

    # =========================================================================
    # 6. Dataset -> 准备一个 Batch 验证 Reload 模型还能正常 Forward
    # =========================================================================
    print("\n========== 6. Dataset ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index in policy.config.action_delta_indices
        ]
    }

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    raw_batch = next(iter(dataloader))

    print(f"state  : {tuple(raw_batch['observation.state'].shape)}")
    print(f"image  : {tuple(raw_batch['observation.images.camera'].shape)}")
    print(f"action : {tuple(raw_batch['action'].shape)}")

    for camera_key in metadata.camera_keys:
        if (
            camera_key in raw_batch
            and isinstance(raw_batch[camera_key], torch.Tensor)
            and raw_batch[camera_key].dtype == torch.uint8
        ):
            raw_batch[camera_key] = (
                raw_batch[camera_key].to(torch.float32)
                / 255.0
            )

    # =========================================================================
    # 7. PreProcessor -> 使用 Checkpoint 自己保存的 Processor 转换 Batch
    # =========================================================================
    print("\n========== 7. PreProcessor ==========")

    batch = preprocessor(raw_batch)

    print(
        "state  :",
        tuple(batch["observation.state"].shape),
        batch["observation.state"].device,
    )

    print(
        "image  :",
        tuple(batch["observation.images.camera"].shape),
        batch["observation.images.camera"].device,
    )

    print(
        "action :",
        tuple(batch["action"].shape),
        batch["action"].device,
    )

    # 保存 Forward 前的 Parameter，确认 Forward 不会修改模型。
    parameter_before = (
        policy.state_dict()[probe_name]
        .detach()
        .clone()
        .cpu()
    )

    # =========================================================================
    # 8. Forward -> 不训练，只验证 Reload 模型可以正常计算
    # =========================================================================
    print("\n========== 8. Reloaded Model Forward ==========")

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            loss, loss_dict = policy.forward(batch)

    loss_value = loss.detach().item()

    print(f"Loss        : {loss_value:.6f}")
    print(f"Loss finite : {torch.isfinite(loss).item()}")

    print("\nLoss details:")

    for key, value in loss_dict.items():
        print(f"  {key}: {value}")

    if not torch.isfinite(loss):
        raise RuntimeError("Reload 模型 Forward 得到 NaN / Inf。")

    # =========================================================================
    # 9. No Training Check -> 确认 Forward 没有修改模型 Parameter
    # =========================================================================
    print("\n========== 9. No Training Check ==========")

    parameter_after = (
        policy.state_dict()[probe_name]
        .detach()
        .clone()
        .cpu()
    )

    parameter_changed = not torch.equal(
        parameter_before,
        parameter_after,
    )

    print(f"Parameter changed : {parameter_changed}")
    print("Expected          : False")

    if parameter_changed:
        raise RuntimeError(
            "本 Demo 没有训练，但 Parameter 发生了变化。"
        )

    # =========================================================================
    # 10. Result -> Checkpoint Reload 验证结果
    # =========================================================================
    print("\n========== 10. Result ==========")

    print("Checkpoint files        : PASS")
    print("Policy reload           : PASS")
    print("PreProcessor reload     : PASS")
    print("PostProcessor reload    : PASS")
    print("Parameter exact match   : PASS")
    print("Forward after reload    : PASS")
    print("Loss finite             : PASS")
    print("Parameter unchanged     : PASS")

    print("\n========== Main Flow ==========")

    print(
        """
Checkpoint on Disk
      ↓
SmolVLAPolicy.from_pretrained()
      ↓
Reload Fine-tuned SmolVLA
      ↓
Reload Pre/PostProcessor
      ↓
Parameter Check
      ↓
Dataset Batch
      ↓
PreProcessor
      ↓
Reloaded SmolVLA
      ↓
Forward
      ↓
Loss
      ↓
Checkpoint OK
"""
    )

    print("========== PASS ==========")
    print("训练后的 SmolVLA Checkpoint 可以正常重新加载和使用。")
    print()
    print("本 Demo 没有执行：")
    print("  loss.backward()")
    print("  optimizer.step()")
    print()
    print("因此模型没有继续训练。")
    print()
    print("下一步：31_smolvla_finetuned_inference.py")


if __name__ == "__main__":
    main()