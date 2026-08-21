# 33_smolvla_prediction_vs_expert.py

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.feature_utils import dataset_to_policy_features


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

BASE_MODEL_ID = "lerobot/smolvla_base"

FINETUNED_CHECKPOINT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/29_smolvla_demo"
)

SAMPLE_INDEX = 5

DEVICE = "cuda"
LOCAL_FILES_ONLY = True


def print_action_chunk(
    title: str,
    chunk: torch.Tensor,
):
    print(f"\n========== {title} ==========")

    chunk_cpu = chunk.detach().float().cpu()

    print(f"Shape : {tuple(chunk_cpu.shape)}")

    actions = chunk_cpu[0]

    for step in range(actions.shape[0]):
        values = ", ".join(
            f"{value:+.6f}"
            for value in actions[step].tolist()
        )

        print(
            f"Action[{step:02d}] = "
            f"[{values}]"
        )


def compute_metrics(
    prediction: torch.Tensor,
    expert: torch.Tensor,
    valid_mask: torch.Tensor,
):
    # prediction / expert:
    # [1, Chunk, ActionDim]
    #
    # valid_mask:
    # [Chunk]

    mask = (
        valid_mask
        .view(1, -1, 1)
        .expand_as(expert)
    )

    difference = prediction - expert

    valid_difference = difference[mask]

    mae = (
        valid_difference
        .abs()
        .mean()
        .item()
    )

    mse = (
        valid_difference
        .pow(2)
        .mean()
        .item()
    )

    rmse = mse ** 0.5

    max_error = (
        valid_difference
        .abs()
        .max()
        .item()
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
    }


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Dataset Metadata -> 获取统一 Feature / Stats
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
    # 2. Fine-tuned Model -> 加载训练后的 Checkpoint
    # =========================================================================
    print("\n========== 2. Load Fine-tuned Model ==========")

    finetuned_policy = SmolVLAPolicy.from_pretrained(
        FINETUNED_CHECKPOINT,
        local_files_only=True,
        strict=True,
    )

    finetuned_policy.eval()

    print(f"Checkpoint : {FINETUNED_CHECKPOINT}")
    print(
        f"Device     : "
        f"{next(finetuned_policy.parameters()).device}"
    )
    print(
        f"Mode       : "
        f"{'train' if finetuned_policy.training else 'eval'}"
    )
    print(
        f"Chunk size : "
        f"{finetuned_policy.config.chunk_size}"
    )
    print(
        f"Action dim : "
        f"{finetuned_policy.config.action_feature.shape[0]}"
    )

    # =========================================================================
    # 3. Fine-tuned Processors
    # =========================================================================
    print("\n========== 3. Load Fine-tuned Processors ==========")

    finetuned_preprocessor = (
        DataProcessorPipeline.from_pretrained(
            FINETUNED_CHECKPOINT,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
    )

    finetuned_postprocessor = (
        DataProcessorPipeline.from_pretrained(
            FINETUNED_CHECKPOINT,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
    )

    print("Fine-tuned PreProcessor  : PASS")
    print("Fine-tuned PostProcessor : PASS")

    # =========================================================================
    # 4. Base Model -> 使用相同 Dataset Action 定义
    # =========================================================================
    print("\n========== 4. Load Base Model ==========")

    base_cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=finetuned_policy.config.chunk_size,
        n_action_steps=finetuned_policy.config.n_action_steps,
        device=DEVICE,
        load_vlm_weights=False,
    )

    base_policy = SmolVLAPolicy.from_pretrained(
        BASE_MODEL_ID,
        config=base_cfg,
        local_files_only=LOCAL_FILES_ONLY,
        strict=False,
    )

    base_policy.eval()

    print(f"Base model : {BASE_MODEL_ID}")
    print(
        f"Chunk size : "
        f"{base_policy.config.chunk_size}"
    )
    print(
        f"Action dim : "
        f"{base_policy.config.action_feature.shape[0]}"
    )

    base_preprocessor, base_postprocessor = (
        make_pre_post_processors(
            policy_cfg=base_cfg,
            dataset_stats=metadata.stats,
        )
    )

    # =========================================================================
    # 5. Dataset -> 读取 Observation + Expert Action Chunk
    # =========================================================================
    print("\n========== 5. Dataset Sample + Expert Chunk ==========")

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index
            in finetuned_policy.config.action_delta_indices
        ]
    }

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    sample = dataset[SAMPLE_INDEX]

    state = sample[
        "observation.state"
    ].clone()

    image = sample[
        "observation.images.camera"
    ].clone()

    task = sample["task"]

    # Dataset 中的 Expert Action 已经是物理 / Dataset Action 空间。
    expert_action_chunk = (
        sample["action"]
        .detach()
        .float()
        .cpu()
        .unsqueeze(0)
    )

    action_is_pad = sample.get(
        "action_is_pad"
    )

    if action_is_pad is None:
        valid_mask = torch.ones(
            expert_action_chunk.shape[1],
            dtype=torch.bool,
        )
    else:
        valid_mask = (
            ~action_is_pad
            .detach()
            .cpu()
            .bool()
        )

    if (
        isinstance(image, torch.Tensor)
        and image.dtype == torch.uint8
    ):
        image = (
            image.to(torch.float32)
            / 255.0
        )

    print(f"Sample index : {SAMPLE_INDEX}")
    print(f"Episode      : {int(sample['episode_index'])}")
    print(f"Frame        : {int(sample['frame_index'])}")
    print(f"Timestamp    : {float(sample['timestamp']):.3f}s")

    print(f"State shape  : {tuple(state.shape)}")
    print(f"Image shape  : {tuple(image.shape)}")
    print(f"Task         : {task!r}")

    print(
        f"Expert Action Chunk : "
        f"{tuple(expert_action_chunk.shape)}"
    )

    print(
        f"Valid Action Steps  : "
        f"{int(valid_mask.sum())} / "
        f"{valid_mask.numel()}"
    )

    # 注意：
    # Expert Action 只用于最后评价，
    # 不允许进入两个模型的推理 Observation。
    print("Expert Action used as model input : False")

    # =========================================================================
    # 6. Same Observation -> Base / Fine-tuned 输入完全相同
    # =========================================================================
    print("\n========== 6. Same Observation ==========")

    base_observation = {
        "observation.state":
            state.clone(),

        "observation.images.camera":
            image.clone(),

        "task":
            task,
    }

    finetuned_observation = {
        "observation.state":
            state.clone(),

        "observation.images.camera":
            image.clone(),

        "task":
            task,
    }

    base_obs = base_preprocessor(
        base_observation
    )

    finetuned_obs = finetuned_preprocessor(
        finetuned_observation
    )

    print("Base PreProcessor       : PASS")
    print("Fine-tuned PreProcessor : PASS")

    # =========================================================================
    # 7. Shared Noise -> 公平比较两个模型
    # =========================================================================
    print("\n========== 7. Shared Noise ==========")

    batch_size = (
        base_obs[
            "observation.state"
        ].shape[0]
    )

    noise_shape = (
        batch_size,
        base_policy.config.chunk_size,
        base_policy.config.max_action_dim,
    )

    device = (
        base_obs[
            "observation.state"
        ].device
    )

    shared_noise = (
        base_policy.model.sample_noise(
            noise_shape,
            device,
        )
    )

    print(
        f"Noise shape  : "
        f"{tuple(shared_noise.shape)}"
    )

    print(
        f"Noise device : "
        f"{shared_noise.device}"
    )

    # =========================================================================
    # 8. Base Model Inference
    # =========================================================================
    print("\n========== 8. Base Model Inference ==========")

    if hasattr(base_policy, "reset"):
        base_policy.reset()

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            base_raw_chunk = (
                base_policy.predict_action_chunk(
                    base_obs,
                    noise=shared_noise.clone(),
                )
            )

    base_action_chunk = (
        base_postprocessor(
            base_raw_chunk
        )
        .detach()
        .float()
        .cpu()
    )

    print(
        f"Base Action Chunk : "
        f"{tuple(base_action_chunk.shape)}"
    )

    # =========================================================================
    # 9. Fine-tuned Model Inference
    # =========================================================================
    print("\n========== 9. Fine-tuned Model Inference ==========")

    if hasattr(finetuned_policy, "reset"):
        finetuned_policy.reset()

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            finetuned_raw_chunk = (
                finetuned_policy.predict_action_chunk(
                    finetuned_obs,
                    noise=shared_noise.clone(),
                )
            )

    finetuned_action_chunk = (
        finetuned_postprocessor(
            {
                "action":
                    finetuned_raw_chunk,
            }
        )["action"]
        .detach()
        .float()
        .cpu()
    )

    print(
        f"Fine-tuned Action Chunk : "
        f"{tuple(finetuned_action_chunk.shape)}"
    )

    # =========================================================================
    # 10. Print Action Chunks
    # =========================================================================
    print_action_chunk(
        "Expert Actions",
        expert_action_chunk,
    )

    print_action_chunk(
        "Base Model Actions",
        base_action_chunk,
    )

    print_action_chunk(
        "Fine-tuned Model Actions",
        finetuned_action_chunk,
    )

    # =========================================================================
    # 11. Overall Error -> Base / Fine-tuned 分别对比 Expert
    # =========================================================================
    print("\n========== 11. Prediction vs Expert ==========")

    base_metrics = compute_metrics(
        base_action_chunk,
        expert_action_chunk,
        valid_mask,
    )

    finetuned_metrics = compute_metrics(
        finetuned_action_chunk,
        expert_action_chunk,
        valid_mask,
    )

    print("\nBase vs Expert:")
    print(
        f"  MAE       : "
        f"{base_metrics['mae']:.8f}"
    )
    print(
        f"  MSE       : "
        f"{base_metrics['mse']:.8f}"
    )
    print(
        f"  RMSE      : "
        f"{base_metrics['rmse']:.8f}"
    )
    print(
        f"  Max Error : "
        f"{base_metrics['max_error']:.8f}"
    )

    print("\nFine-tuned vs Expert:")
    print(
        f"  MAE       : "
        f"{finetuned_metrics['mae']:.8f}"
    )
    print(
        f"  MSE       : "
        f"{finetuned_metrics['mse']:.8f}"
    )
    print(
        f"  RMSE      : "
        f"{finetuned_metrics['rmse']:.8f}"
    )
    print(
        f"  Max Error : "
        f"{finetuned_metrics['max_error']:.8f}"
    )

    # =========================================================================
    # 12. Improvement -> Fine-tuned 是否更接近 Expert
    # =========================================================================
    print("\n========== 12. Fine-tuning Improvement ==========")

    mae_improvement = (
        base_metrics["mae"]
        - finetuned_metrics["mae"]
    )

    rmse_improvement = (
        base_metrics["rmse"]
        - finetuned_metrics["rmse"]
    )

    if base_metrics["mae"] > 0:
        mae_improvement_percent = (
            mae_improvement
            / base_metrics["mae"]
            * 100.0
        )
    else:
        mae_improvement_percent = 0.0

    finetuned_better = (
        finetuned_metrics["mae"]
        < base_metrics["mae"]
    )

    print(
        f"MAE improvement       : "
        f"{mae_improvement:+.8f}"
    )

    print(
        f"MAE improvement ratio : "
        f"{mae_improvement_percent:+.2f}%"
    )

    print(
        f"RMSE improvement      : "
        f"{rmse_improvement:+.8f}"
    )

    print(
        f"Fine-tuned better     : "
        f"{finetuned_better}"
    )

    # =========================================================================
    # 13. Per-step Error
    # =========================================================================
    print("\n========== 13. Per-step Error ==========")

    base_abs_error = (
        base_action_chunk
        - expert_action_chunk
    ).abs()

    finetuned_abs_error = (
        finetuned_action_chunk
        - expert_action_chunk
    ).abs()

    for step in range(
        expert_action_chunk.shape[1]
    ):
        if not bool(valid_mask[step]):
            print(
                f"Action[{step:02d}] | PAD"
            )
            continue

        base_step_mae = (
            base_abs_error[
                0,
                step,
            ]
            .mean()
            .item()
        )

        finetuned_step_mae = (
            finetuned_abs_error[
                0,
                step,
            ]
            .mean()
            .item()
        )

        better = (
            "Fine-tuned"
            if finetuned_step_mae < base_step_mae
            else "Base"
        )

        print(
            f"Action[{step:02d}] | "
            f"Base MAE={base_step_mae:.8f} | "
            f"Fine-tuned MAE={finetuned_step_mae:.8f} | "
            f"Better={better}"
        )

    # =========================================================================
    # 14. Per-dimension Error
    # =========================================================================
    print("\n========== 14. Per-dimension Error ==========")

    for dim in range(
        expert_action_chunk.shape[2]
    ):
        base_joint_error = (
            base_abs_error[
                0,
                valid_mask,
                dim,
            ]
        )

        finetuned_joint_error = (
            finetuned_abs_error[
                0,
                valid_mask,
                dim,
            ]
        )

        base_joint_mae = (
            base_joint_error
            .mean()
            .item()
        )

        finetuned_joint_mae = (
            finetuned_joint_error
            .mean()
            .item()
        )

        better = (
            "Fine-tuned"
            if finetuned_joint_mae < base_joint_mae
            else "Base"
        )

        print(
            f"Joint[{dim + 1}] | "
            f"Base MAE={base_joint_mae:.8f} | "
            f"Fine-tuned MAE={finetuned_joint_mae:.8f} | "
            f"Better={better}"
        )

    # =========================================================================
    # 15. Result
    # =========================================================================
    print("\n========== 15. Result ==========")

    print("Same Observation         : PASS")
    print("Same Flow Matching Noise : PASS")
    print("Expert Action excluded   : PASS")
    print("Base inference           : PASS")
    print("Fine-tuned inference     : PASS")
    print("Expert comparison        : PASS")

    print()

    print(
        f"Base MAE       : "
        f"{base_metrics['mae']:.8f}"
    )

    print(
        f"Fine-tuned MAE : "
        f"{finetuned_metrics['mae']:.8f}"
    )

    print(
        f"Fine-tuned better : "
        f"{finetuned_better}"
    )

    # =========================================================================
    # 16. Main Flow
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
                  Same Observation
             Camera + State + Task
                       │
                 Same Noise
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
      smolvla_base           Fine-tuned
            │                     │
            ▼                     ▼
     Base Prediction      Fine-tuned Prediction
            │                     │
            └──────────┬──────────┘
                       │
                       ▼
              Expert Action Chunk
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Base vs Expert   Fine-tuned vs Expert
              │                 │
              └────────┬────────┘
                       ▼
                 Compare Error
"""
    )

    print("========== DONE ==========")

    print()
    print("32 回答：")
    print("Fine-tuning 有没有改变模型输出？")
    print()
    print("33 回答：")
    print("Fine-tuning 后的输出有没有更接近 Expert Action？")


if __name__ == "__main__":
    main()