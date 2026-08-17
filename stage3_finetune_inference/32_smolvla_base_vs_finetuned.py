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


def print_action_chunk(title: str, chunk: torch.Tensor):
    print(f"\n========== {title} ==========")

    chunk_cpu = chunk.detach().cpu()

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
    # 1. Dataset Metadata -> 用 Dataset 定义 Base Model 的输入 / 输出
    # =========================================================================
    print("========== 1. Dataset Metadata ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

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

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    # =========================================================================
    # 2. Load Fine-tuned Model -> 训练后的模型
    # =========================================================================
    print("\n========== 2. Load Fine-tuned Model ==========")

    finetuned_policy = SmolVLAPolicy.from_pretrained(
        FINETUNED_CHECKPOINT,
        local_files_only=True,
        strict=True,
    )

    finetuned_policy.eval()

    print(f"Checkpoint : {FINETUNED_CHECKPOINT}")
    print(f"Chunk size : {finetuned_policy.config.chunk_size}")
    print(
        f"Action dim : "
        f"{finetuned_policy.config.action_feature.shape[0]}"
    )

    # =========================================================================
    # 3. Load Fine-tuned Processor
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
    # 4. Build Base Model -> 使用与 Fine-tuned 相同的 Feature / Chunk 定义
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
    print(f"Chunk size : {base_policy.config.chunk_size}")
    print(
        f"Action dim : "
        f"{base_policy.config.action_feature.shape[0]}"
    )

    # Base Model 必须使用同一个 Dataset 的 stats。
    base_preprocessor, base_postprocessor = (
        make_pre_post_processors(
            policy_cfg=base_cfg,
            dataset_stats=metadata.stats,
        )
    )

    # =========================================================================
    # 5. Observation -> 同一个输入同时给两个模型
    # =========================================================================
    print("\n========== 5. Build Same Observation ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    sample = dataset[SAMPLE_INDEX]

    observation = {
        "observation.state":
            sample["observation.state"].clone(),

        "observation.images.camera":
            sample["observation.images.camera"].clone(),

        "task":
            sample["task"],
    }

    camera_key = "observation.images.camera"

    if observation[camera_key].dtype == torch.uint8:
        observation[camera_key] = (
            observation[camera_key].to(torch.float32)
            / 255.0
        )

    print(f"Sample index : {SAMPLE_INDEX}")
    print(f"Episode      : {int(sample['episode_index'])}")
    print(f"Frame        : {int(sample['frame_index'])}")
    print(f"Timestamp    : {float(sample['timestamp']):.3f}s")

    print(
        f"State shape  : "
        f"{tuple(observation['observation.state'].shape)}"
    )

    print(
        f"Image shape  : "
        f"{tuple(observation[camera_key].shape)}"
    )

    print(f"Task         : {observation['task']!r}")
    print(f"Expert Action included : {'action' in observation}")

    # =========================================================================
    # 6. PreProcessor -> 分别转换，但原始 Observation 完全相同
    # =========================================================================
    print("\n========== 6. PreProcessor ==========")

    base_obs = base_preprocessor(
        {
            "observation.state":
                observation["observation.state"].clone(),

            "observation.images.camera":
                observation[camera_key].clone(),

            "task":
                observation["task"],
        }
    )

    finetuned_obs = finetuned_preprocessor(
        {
            "observation.state":
                observation["observation.state"].clone(),

            "observation.images.camera":
                observation[camera_key].clone(),

            "task":
                observation["task"],
        }
    )

    print("Base PreProcessor       : PASS")
    print("Fine-tuned PreProcessor : PASS")

    # =========================================================================
    # 7. Base Inference
    # =========================================================================
    print("\n========== 7. Base Model Inference ==========")

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
                    base_obs
                )
            )

    base_action_chunk = base_postprocessor(
        {
            "action": base_raw_chunk,
        }
    )["action"]

    print(
        f"Base Action Chunk : "
        f"{tuple(base_action_chunk.shape)}"
    )

    # =========================================================================
    # 8. Fine-tuned Inference
    # =========================================================================
    print("\n========== 8. Fine-tuned Model Inference ==========")

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
                    finetuned_obs
                )
            )

    finetuned_action_chunk = (
        finetuned_postprocessor(
            {
                "action": finetuned_raw_chunk,
            }
        )["action"]
    )

    print(
        f"Fine-tuned Action Chunk : "
        f"{tuple(finetuned_action_chunk.shape)}"
    )

    # =========================================================================
    # 9. Print Action Chunk -> 查看两边实际输出
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
    # 10. Difference -> 比较训练前后模型行为
    # =========================================================================
    print("\n========== 10. Base vs Fine-tuned ==========")

    difference = (
        finetuned_action_chunk.detach().float().cpu()
        - base_action_chunk.detach().float().cpu()
    )

    abs_difference = difference.abs()

    print(
        f"Mean |difference| : "
        f"{abs_difference.mean().item():.8f}"
    )

    print(
        f"Max  |difference| : "
        f"{abs_difference.max().item():.8f}"
    )

    print(
        f"L2 difference     : "
        f"{difference.norm().item():.8f}"
    )

    same_output = torch.equal(
        base_action_chunk.detach().cpu(),
        finetuned_action_chunk.detach().cpu(),
    )

    print(f"Exactly same      : {same_output}")
    print(f"Model behavior changed : {not same_output}")

    # =========================================================================
    # 11. Per-step Difference
    # =========================================================================
    print("\n========== 11. Per-step Difference ==========")

    for step in range(difference.shape[1]):
        step_diff = abs_difference[0, step]

        print(
            f"Action[{step:02d}] | "
            f"mean diff={step_diff.mean().item():.8f} | "
            f"max diff={step_diff.max().item():.8f}"
        )

    # =========================================================================
    # 12. Result
    # =========================================================================
    print("\n========== 12. Result ==========")

    print("Same Observation         : PASS")
    print("Base inference           : PASS")
    print("Fine-tuned inference     : PASS")
    print("Base PostProcessor       : PASS")
    print("Fine-tuned PostProcessor : PASS")
    print("Action comparison        : PASS")

    print()
    print(
        f"Model behavior changed : "
        f"{not same_output}"
    )

    # =========================================================================
    # 13. 主干总结
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
                  Same Observation
             Camera + State + Task
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       smolvla_base         Fine-tuned
             │                   │
             ▼                   ▼
        PreProcessor         PreProcessor
             │                   │
             ▼                   ▼
     predict_action_chunk  predict_action_chunk
             │                   │
             ▼                   ▼
        PostProcessor        PostProcessor
             │                   │
             ▼                   ▼
       Base Actions       Fine-tuned Actions
             │                   │
             └─────────┬─────────┘
                       ▼
                 Compare Difference
"""
    )

    print("========== PASS ==========")

    print()
    print("本 Demo 只回答：")
    print()
    print(
        "Fine-tuning 是否改变了模型在同一个 "
        "Observation 下输出的 Action？"
    )
    print()
    print("它暂时不回答：")
    print()
    print(
        "Fine-tuned Action 是否比 Base Action 更正确。"
    )
    print()
    print(
        "下一步：33_smolvla_prediction_vs_expert.py"
    )


if __name__ == "__main__":
    main()