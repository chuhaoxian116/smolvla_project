# 32_smolvla_base_vs_finetuned.py

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

    # [1, Chunk, ActionDim] -> [Chunk, ActionDim]
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


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Dataset Metadata -> 获取统一的 Feature / Stats
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
    # 2. Fine-tuned Model -> 加载 Stage 2 训练后的 Checkpoint
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
    print(
        f"Max action dim : "
        f"{finetuned_policy.config.max_action_dim}"
    )

    # =========================================================================
    # 3. Fine-tuned Processors -> 恢复训练时保存的 Processor
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
    # 4. Base Model -> 使用和 Fine-tuned 完全相同的 Action 定义
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
        f"Device     : "
        f"{next(base_policy.parameters()).device}"
    )
    print(
        f"Mode       : "
        f"{'train' if base_policy.training else 'eval'}"
    )
    print(
        f"Chunk size : "
        f"{base_policy.config.chunk_size}"
    )
    print(
        f"Action dim : "
        f"{base_policy.config.action_feature.shape[0]}"
    )
    print(
        f"Max action dim : "
        f"{base_policy.config.max_action_dim}"
    )

    # Base Model 使用相同 Dataset Stats，
    # 保证输入 / 输出处于同一数据空间。
    base_preprocessor, base_postprocessor = (
        make_pre_post_processors(
            policy_cfg=base_cfg,
            dataset_stats=metadata.stats,
        )
    )

    # =========================================================================
    # 5. Config Check -> 两个模型必须具有相同 Action 定义
    # =========================================================================
    print("\n========== 5. Config Check ==========")

    same_chunk_size = (
        base_policy.config.chunk_size
        == finetuned_policy.config.chunk_size
    )

    same_action_dim = (
        base_policy.config.action_feature.shape[0]
        == finetuned_policy.config.action_feature.shape[0]
    )

    same_max_action_dim = (
        base_policy.config.max_action_dim
        == finetuned_policy.config.max_action_dim
    )

    print(f"Same chunk size     : {same_chunk_size}")
    print(f"Same action dim     : {same_action_dim}")
    print(f"Same max action dim : {same_max_action_dim}")

    if not (
        same_chunk_size
        and same_action_dim
        and same_max_action_dim
    ):
        raise RuntimeError(
            "Base / Fine-tuned Action Config 不一致。"
        )

    # =========================================================================
    # 6. Same Observation -> 两个模型使用完全相同的输入
    # =========================================================================
    print("\n========== 6. Build Same Observation ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    sample = dataset[SAMPLE_INDEX]

    state = sample[
        "observation.state"
    ].clone()

    image = sample[
        "observation.images.camera"
    ].clone()

    task = sample["task"]

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

    # 推理阶段不提供 Expert Action。
    print("Expert Action included : False")

    # =========================================================================
    # 7. PreProcessor -> 同一个 Observation 分别转换
    # =========================================================================
    print("\n========== 7. PreProcessor ==========")

    base_observation = {
        "observation.state": state.clone(),
        "observation.images.camera": image.clone(),
        "task": task,
    }

    finetuned_observation = {
        "observation.state": state.clone(),
        "observation.images.camera": image.clone(),
        "task": task,
    }

    base_obs = base_preprocessor(
        base_observation
    )

    finetuned_obs = finetuned_preprocessor(
        finetuned_observation
    )

    print("Base PreProcessor       : PASS")
    print("Fine-tuned PreProcessor : PASS")

    print(
        "Base state      :",
        tuple(base_obs["observation.state"].shape),
        base_obs["observation.state"].device,
    )

    print(
        "Fine-tuned state:",
        tuple(
            finetuned_obs[
                "observation.state"
            ].shape
        ),
        finetuned_obs[
            "observation.state"
        ].device,
    )

    # =========================================================================
    # 8. Shared Noise -> 两个模型使用完全相同的 Flow Matching 初始 Noise
    # =========================================================================
    print("\n========== 8. Shared Noise ==========")

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

    base_device = (
        base_obs[
            "observation.state"
        ].device
    )

    # SmolVLA 如果不指定 noise，会在每次推理时重新随机采样。
    # 为了公平比较 Base / Fine-tuned，
    # 这里显式生成一份 Noise，并同时提供给两个模型。
    shared_noise = (
        base_policy.model.sample_noise(
            noise_shape,
            base_device,
        )
    )

    print(f"Noise shape  : {tuple(shared_noise.shape)}")
    print(f"Noise dtype  : {shared_noise.dtype}")
    print(f"Noise device : {shared_noise.device}")

    # =========================================================================
    # 9. Base Model Inference
    # =========================================================================
    print("\n========== 9. Base Model Inference ==========")

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

    print(
        f"Base Raw Action Chunk : "
        f"{tuple(base_raw_chunk.shape)}"
    )

    # make_pre_post_processors() 创建出的 PostProcessor
    # 当前接口直接接收 PolicyAction Tensor。
    base_action_chunk = (
        base_postprocessor(
            base_raw_chunk
        )
    )

    print(
        f"Base Action Chunk     : "
        f"{tuple(base_action_chunk.shape)}"
    )

    # =========================================================================
    # 10. Fine-tuned Model Inference
    # =========================================================================
    print("\n========== 10. Fine-tuned Model Inference ==========")

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

    print(
        f"Fine-tuned Raw Action Chunk : "
        f"{tuple(finetuned_raw_chunk.shape)}"
    )

    # 当前从 checkpoint reload 的 PostProcessor，
    # 31 已经验证需要 Transition 字典输入。
    finetuned_action_chunk = (
        finetuned_postprocessor(
            {
                "action":
                    finetuned_raw_chunk,
            }
        )["action"]
    )

    print(
        f"Fine-tuned Action Chunk     : "
        f"{tuple(finetuned_action_chunk.shape)}"
    )

    # =========================================================================
    # 11. Action Shape Check
    # =========================================================================
    print("\n========== 11. Action Shape Check ==========")

    base_shape = tuple(
        base_action_chunk.shape
    )

    finetuned_shape = tuple(
        finetuned_action_chunk.shape
    )

    expected_shape = (
        batch_size,
        base_policy.config.chunk_size,
        base_policy.config.action_feature.shape[0],
    )

    print(f"Base shape       : {base_shape}")
    print(f"Fine-tuned shape : {finetuned_shape}")
    print(f"Expected shape   : {expected_shape}")

    shape_ok = (
        base_shape == expected_shape
        and finetuned_shape == expected_shape
    )

    print(
        f"Shape check      : "
        f"{'PASS' if shape_ok else 'FAIL'}"
    )

    if not shape_ok:
        raise RuntimeError(
            "Base / Fine-tuned Action Shape 异常。"
        )

    # =========================================================================
    # 12. Print Actions
    # =========================================================================
    print_action_chunk(
        "Base Model Actions",
        base_action_chunk,
    )

    print_action_chunk(
        "Fine-tuned Model Actions",
        finetuned_action_chunk,
    )

    # =========================================================================
    # 13. Difference -> 训练前后 Action 差异
    # =========================================================================
    print("\n========== 13. Base vs Fine-tuned ==========")

    base_cpu = (
        base_action_chunk
        .detach()
        .float()
        .cpu()
    )

    finetuned_cpu = (
        finetuned_action_chunk
        .detach()
        .float()
        .cpu()
    )

    difference = (
        finetuned_cpu
        - base_cpu
    )

    abs_difference = (
        difference.abs()
    )

    mean_difference = (
        abs_difference
        .mean()
        .item()
    )

    max_difference = (
        abs_difference
        .max()
        .item()
    )

    l2_difference = (
        difference
        .norm()
        .item()
    )

    exactly_same = torch.equal(
        base_cpu,
        finetuned_cpu,
    )

    print(
        f"Mean |difference| : "
        f"{mean_difference:.8f}"
    )

    print(
        f"Max  |difference| : "
        f"{max_difference:.8f}"
    )

    print(
        f"L2 difference     : "
        f"{l2_difference:.8f}"
    )

    print(
        f"Exactly same      : "
        f"{exactly_same}"
    )

    print(
        f"Model behavior changed : "
        f"{not exactly_same}"
    )

    # =========================================================================
    # 14. Per-step Difference
    # =========================================================================
    print("\n========== 14. Per-step Difference ==========")

    for step in range(
        difference.shape[1]
    ):
        step_diff = (
            abs_difference[
                0,
                step,
            ]
        )

        print(
            f"Action[{step:02d}] | "
            f"mean diff="
            f"{step_diff.mean().item():.8f} | "
            f"max diff="
            f"{step_diff.max().item():.8f}"
        )

    # =========================================================================
    # 15. Per-dimension Difference
    # =========================================================================
    print("\n========== 15. Per-dimension Difference ==========")

    action_dim = (
        difference.shape[2]
    )

    for dim in range(action_dim):
        dim_diff = (
            abs_difference[
                :,
                :,
                dim,
            ]
        )

        print(
            f"Joint[{dim + 1}] | "
            f"mean diff="
            f"{dim_diff.mean().item():.8f} | "
            f"max diff="
            f"{dim_diff.max().item():.8f}"
        )

    # =========================================================================
    # 16. Result
    # =========================================================================
    print("\n========== 16. Result ==========")

    print("Same Observation         : PASS")
    print("Same Flow Matching Noise : PASS")
    print("Same Action Config       : PASS")
    print("Base inference           : PASS")
    print("Fine-tuned inference     : PASS")
    print("Base PostProcessor       : PASS")
    print("Fine-tuned PostProcessor : PASS")
    print("Action shape             : PASS")
    print("Action comparison        : PASS")

    print()

    print(
        f"Model behavior changed : "
        f"{not exactly_same}"
    )

    # =========================================================================
    # 17. Main Flow
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
                   Same Observation
              Camera + State + Task
                         │
                         │
                   Same Noise
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       smolvla_base             Fine-tuned
             │                       │
             ▼                       ▼
    predict_action_chunk     predict_action_chunk
             │                       │
             ▼                       ▼
       PostProcessor             PostProcessor
             │                       │
             ▼                       ▼
       Base Actions           Fine-tuned Actions
             │                       │
             └───────────┬───────────┘
                         ▼
                 Compare Difference
"""
    )

    print("========== PASS ==========")

    print()
    print("本 Demo 控制了两个主要变量：")
    print()
    print("  1. Same Observation")
    print("  2. Same Flow Matching Noise")
    print()
    print("因此两边最主要的差异来自：")
    print()
    print("  Base Parameter")
    print("       vs")
    print("  Fine-tuned Parameter")
    print()
    print("本 Demo 只回答：")
    print()
    print(
        "Fine-tuning 是否改变了模型在相同条件下"
        "生成的 Action？"
    )
    print()
    print("它暂时不回答：")
    print()
    print(
        "Fine-tuned Action 是否更接近 Expert Action。"
    )
    print()
    print(
        "下一步："
        "33_smolvla_prediction_vs_expert.py"
    )


if __name__ == "__main__":
    main()