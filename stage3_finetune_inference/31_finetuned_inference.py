# 31_smolvla_finetuned_inference.py

from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset
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

SAMPLE_INDEX = 5
DEVICE = "cuda"


def print_tensor(name: str, value):
    if isinstance(value, torch.Tensor):
        print(
            f"{name:<36} "
            f"shape={tuple(value.shape)!s:<16} "
            f"dtype={str(value.dtype):<15} "
            f"device={value.device}"
        )
    else:
        print(f"{name:<36} type={type(value).__name__}")


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Reload Checkpoint -> 加载训练后的模型
    # =========================================================================
    print("========== 1. Reload Fine-tuned SmolVLA ==========")

    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_DIR,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    print(f"Checkpoint   : {CHECKPOINT_DIR}")
    print(f"Policy device: {next(policy.parameters()).device}")
    print(f"Policy mode  : {'train' if policy.training else 'eval'}")
    print(f"Chunk size   : {policy.config.chunk_size}")
    print(f"Action dim   : {policy.config.action_feature.shape[0]}")

    # =========================================================================
    # 2. Reload Processor -> 恢复训练时的输入 / 输出处理
    # =========================================================================
    print("\n========== 2. Reload Processors ==========")

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
    # 3. Dataset Frame -> 只拿 Observation，不给模型 Expert Action
    # =========================================================================
    print("\n========== 3. Build Inference Observation ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    sample = dataset[SAMPLE_INDEX]

    print(f"Sample index : {SAMPLE_INDEX}")
    print(f"Episode      : {int(sample['episode_index'])}")
    print(f"Frame        : {int(sample['frame_index'])}")
    print(f"Timestamp    : {float(sample['timestamp']):.3f}s")

    # 推理阶段只提供：
    # Camera + State + Task
    observation = {
        "observation.state":
            sample["observation.state"],

        "observation.images.camera":
            sample["observation.images.camera"],

        "task":
            sample["task"],
    }

    # 与训练脚本保持一致：
    # Camera 如果还是 uint8，则转成 [0, 1] float32。
    camera_key = "observation.images.camera"

    if (
        isinstance(observation[camera_key], torch.Tensor)
        and observation[camera_key].dtype == torch.uint8
    ):
        observation[camera_key] = (
            observation[camera_key].to(torch.float32)
            / 255.0
        )

    print("\nInference input:")

    for key, value in observation.items():
        print_tensor(key, value)

    print()
    print(f"Contains Expert Action : {'action' in observation}")

    if "action" in observation:
        raise RuntimeError(
            "Inference Observation 不应该包含 Expert Action。"
        )

    # =========================================================================
    # 4. PreProcessor -> Observation 转成 SmolVLA 输入
    # =========================================================================
    print("\n========== 4. PreProcessor ==========")

    processed_obs = preprocessor(observation)

    for key, value in processed_obs.items():
        print_tensor(key, value)

    # =========================================================================
    # 5. Fine-tuned SmolVLA -> 预测完整 Action Chunk
    # =========================================================================
    print("\n========== 5. Predict Action Chunk ==========")

    if hasattr(policy, "reset"):
        policy.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(DEVICE == "cuda"),
        ):
            predicted_chunk = policy.predict_action_chunk(
                processed_obs
            )

    print_tensor(
        "predicted action chunk",
        predicted_chunk,
    )

    print("\nRaw model output:")
    print(predicted_chunk)

    # =========================================================================
    # 6. PostProcessor -> 恢复到 Dataset Action 的物理数据空间
    # =========================================================================
    print("\n========== 6. PostProcessor ==========")

    # 当前 LeRobot Processor Pipeline 接收的是字典形式的 Transition。
    # 将 Policy 输出放入 "action"，PostProcessor 完成反归一化后再取出。
    action_chunk = postprocessor(
        {
            "action": predicted_chunk,
        }
    )["action"]

    print_tensor(
        "postprocessed action chunk",
        action_chunk,
    )

    print("\nPostprocessed Action Chunk:")
    print(action_chunk)

    # =========================================================================
    # 7. Inspect Actions -> 查看未来每一步预测
    # =========================================================================
    print("\n========== 7. Predicted Actions ==========")

    # predict_action_chunk:
    # [Batch, Chunk, ActionDim]
    #
    # 当前：
    # [1, 10, 6]
    chunk = action_chunk[0]

    for step in range(chunk.shape[0]):
        values = chunk[step].detach().cpu().tolist()

        values_text = ", ".join(
            f"{value:+.6f}"
            for value in values
        )

        print(
            f"Action[{step:02d}] = "
            f"[{values_text}]"
        )

    # =========================================================================
    # 8. Shape Check -> 验证 Checkpoint 的输出定义
    # =========================================================================
    print("\n========== 8. Shape Check ==========")

    expected_chunk_size = policy.config.chunk_size
    expected_action_dim = (
        policy.config.action_feature.shape[0]
    )

    actual_shape = tuple(action_chunk.shape)

    print(f"Actual shape   : {actual_shape}")
    print(
        f"Expected shape : "
        f"(1, {expected_chunk_size}, {expected_action_dim})"
    )

    shape_ok = (
        action_chunk.ndim == 3
        and action_chunk.shape[0] == 1
        and action_chunk.shape[1] == expected_chunk_size
        and action_chunk.shape[2] == expected_action_dim
    )

    print(f"Shape check    : {'PASS' if shape_ok else 'FAIL'}")

    if not shape_ok:
        raise RuntimeError(
            f"Unexpected Action Chunk shape: {actual_shape}"
        )

    # =========================================================================
    # 9. Result
    # =========================================================================
    print("\n========== 9. Result ==========")

    print("Checkpoint reload       : PASS")
    print("Observation only        : PASS")
    print("Expert Action excluded  : PASS")
    print("PreProcessor            : PASS")
    print("Fine-tuned inference    : PASS")
    print("Action Chunk            : PASS")
    print("PostProcessor           : PASS")
    print("Action shape            : PASS")

    if torch.cuda.is_available():
        peak_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"CUDA peak allocated     : "
            f"{peak_gb:.3f} GiB"
        )

    # =========================================================================
    # 10. 主干总结
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
Fine-tuned Checkpoint
        ↓
Reload SmolVLA
        ↓
Camera + State + Task
        ↓
PreProcessor
        ↓
Fine-tuned SmolVLA
        ↓
predict_action_chunk()
        ↓
Predicted Action Chunk
        ↓
PostProcessor
        ↓
Physical Action Space
        ↓
[1, ChunkSize, ActionDim]
"""
    )

    print("========== PASS ==========")

    print()
    print("Inference 阶段没有提供 Expert Action。")
    print()
    print(
        "当前 Action 的语义来自训练 Dataset："
    )
    print(
        "6 维绝对 Joint Target。"
    )
    print()
    print(
        "下一步：32_smolvla_base_vs_finetuned.py"
    )


if __name__ == "__main__":
    main()