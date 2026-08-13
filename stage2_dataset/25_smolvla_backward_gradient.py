# 25_smolvla_backward_gradient.py

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

LOCAL_FILES_ONLY = True

# 最多打印多少个有 Gradient 的参数。
SHOW_GRAD_PARAMS = 15


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

    # LeRobot 官方训练流程中，Camera 输入需要转成 [0, 1] float32。
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

    print("\n========== 6. Trainable Parameters ==========")

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    first_trainable_name = None
    first_trainable_param = None

    for name, param in policy.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

            if first_trainable_param is None:
                first_trainable_name = name
                first_trainable_param = param
        else:
            frozen_params += param.numel()

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Frozen parameters    : {frozen_params:,}")

    if first_trainable_param is None:
        raise RuntimeError("没有找到 requires_grad=True 的参数。")

    # 只保存一个标量，用来验证 backward 前后参数本身不会发生变化。
    parameter_value_before = (
        first_trainable_param.detach()
        .reshape(-1)[0]
        .float()
        .item()
    )

    print("\nFirst trainable parameter:")
    print(f"  name  : {first_trainable_name}")
    print(f"  shape : {tuple(first_trainable_param.shape)}")
    print(f"  value : {parameter_value_before:.10f}")

    print("\n========== 7. Before Forward ==========")

    # 清除以前可能残留的 Gradient。
    #
    # 注意：
    # 这里虽然还没有 Optimizer，
    # 但 Parameter.grad 本身仍然需要清空。
    policy.zero_grad(set_to_none=True)

    grad_count_before = sum(
        1
        for param in policy.parameters()
        if param.grad is not None
    )

    print(f"Parameters with grad before forward : {grad_count_before}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n========== 8. SmolVLA Forward ==========")

    # 注意：
    # 与 24 号代码相比，这里已经删除：
    #
    # with torch.no_grad():
    #
    # 因为 backward() 必须依赖 Forward 过程中建立的计算图。
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(DEVICE == "cuda"),
    ):
        loss, loss_dict = policy.forward(processed_batch)

    print(f"loss       : {float(loss):.6f}")
    print(f"loss dtype : {loss.dtype}")
    print(f"loss device: {loss.device}")
    print(f"requires_grad: {loss.requires_grad}")

    print("\nLoss details:")

    for key, value in loss_dict.items():
        print(f"  {key}: {value}")

    grad_count_after_forward = sum(
        1
        for param in policy.parameters()
        if param.grad is not None
    )

    print(
        "\nParameters with grad after forward : "
        f"{grad_count_after_forward}"
    )

    print()
    print("说明：")
    print("Forward 只建立计算图并得到 Loss。")
    print("此时 Parameter.grad 仍然没有被计算。")

    print("\n========== 9. loss.backward() ==========")

    loss.backward()

    print("loss.backward() completed.")

    print("\n========== 10. Gradient Check ==========")

    grad_param_count = 0
    no_grad_param_count = 0

    total_grad_squared = 0.0

    shown = 0

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            no_grad_param_count += 1
            continue

        grad_param_count += 1

        grad = param.grad.detach()

        grad_norm = grad.float().norm().item()
        grad_abs_mean = grad.float().abs().mean().item()
        grad_abs_max = grad.float().abs().max().item()

        total_grad_squared += grad_norm ** 2

        if shown < SHOW_GRAD_PARAMS:
            print(
                f"{name}\n"
                f"  param shape : {tuple(param.shape)}\n"
                f"  grad shape  : {tuple(grad.shape)}\n"
                f"  grad norm   : {grad_norm:.8e}\n"
                f"  |grad| mean : {grad_abs_mean:.8e}\n"
                f"  |grad| max  : {grad_abs_max:.8e}"
            )

            shown += 1

    global_grad_norm = total_grad_squared ** 0.5

    print("\n========== 11. Gradient Summary ==========")

    print(
        f"Trainable params with Gradient : "
        f"{grad_param_count}"
    )

    print(
        f"Trainable params without grad  : "
        f"{no_grad_param_count}"
    )

    print(
        f"Global Gradient L2 norm        : "
        f"{global_grad_norm:.8f}"
    )

    print("\n========== 12. Parameter Change Check ==========")

    parameter_value_after_backward = (
        first_trainable_param.detach()
        .reshape(-1)[0]
        .float()
        .item()
    )

    print(f"Parameter : {first_trainable_name}")
    print(f"Before backward : {parameter_value_before:.10f}")
    print(f"After backward  : {parameter_value_after_backward:.10f}")

    changed = parameter_value_before != parameter_value_after_backward

    print(f"Parameter changed : {changed}")

    print()
    print("Expected:")
    print("Parameter changed = False")
    print()
    print("因为：")
    print("loss.backward() 只计算 Gradient。")
    print("它不会直接修改模型参数。")

    if first_trainable_param.grad is not None:
        first_grad = (
            first_trainable_param.grad.detach()
            .reshape(-1)[0]
            .float()
            .item()
        )

        print()
        print("First parameter gradient:")
        print(f"  grad[0] = {first_grad:.10e}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(
            f"\nCUDA peak allocated after backward: "
            f"{peak_gb:.3f} GiB"
        )

    print("\n========== 13. Final Concept ==========")

    print(
        """
当前已经完成：

Dataset
   ↓
DataLoader
   ↓
PreProcessor
   ↓
SmolVLA Forward
   ↓
Loss
   ↓
loss.backward()
   ↓
Gradient
   ↓
Parameter.grad

但是：

Parameter
   ↓
没有变化

原因：

loss.backward()
=
根据 Loss 计算每一个可训练参数的 Gradient

它只回答：

    这个参数应该往哪个方向变化？
    应该变化多大？

它不会真正修改 Parameter。

真正修改参数需要下一步：

    optimizer.step()

因此：

Forward
    =
计算 Loss

Backward
    =
计算 Gradient

Optimizer.step()
    =
根据 Gradient 真正更新 Parameter
"""
    )

    print("========== 14. PASS ==========")

    print("SmolVLA Forward   : PASS")
    print("Loss              : PASS")
    print("Backward          : PASS")
    print("Gradient          : PASS")
    print("Parameter Update  : NOT EXECUTED")
    print()
    print("下一步：26_smolvla_optimizer_step.py")


if __name__ == "__main__":
    main()