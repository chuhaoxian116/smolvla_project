import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"


def main():
    print("===== 1. Load Model =====")

    # 加载 SmolVLA
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    # 创建输入、输出 Processor
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    print("\n===== 2. Build Fake Observation =====")

    fake_observation = {
        "observation.state": torch.tensor(
            [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
            dtype=torch.float32,
        ),

        "observation.images.camera1": torch.zeros(
            3, 256, 256, dtype=torch.float32
        ),

        "observation.images.camera2": torch.zeros(
            3, 256, 256, dtype=torch.float32
        ),

        "observation.images.camera3": torch.zeros(
            3, 256, 256, dtype=torch.float32
        ),

        "task": "pick up the red cube",
    }

    print("\n===== 3. PreProcess =====")

    # Raw Observation -> 模型输入 Tensor
    processed = preprocessor(fake_observation)

    print("State :", processed["observation.state"].shape)
    print("Camera:", processed["observation.images.camera1"].shape)
    print("Tokens:", processed["observation.language.tokens"].shape)

    print("\n===== 4. SmolVLA Predict =====")

    # SmolVLA 一次生成 50 个 Action
    with torch.inference_mode():
        model_action_chunk = policy.predict_action_chunk(processed)

    print("Model Action:")
    print("  shape :", tuple(model_action_chunk.shape))
    print("  dtype :", model_action_chunk.dtype)
    print("  device:", model_action_chunk.device)

    print("\n===== 5. PostProcess =====")

    # 模型 Action -> PostProcessor
    # PostProcessor 负责：
    # 1. Unnormalize（有对应 stats 时）
    # 2. CUDA -> CPU
    robot_action_chunk = postprocessor(model_action_chunk)

    print("Postprocessed Action:")
    print("  shape :", tuple(robot_action_chunk.shape))
    print("  dtype :", robot_action_chunk.dtype)
    print("  device:", robot_action_chunk.device)

    print("\n===== 6. Compare =====")

    # 对比第一个 Action
    print("Model Action[00]:")
    print(model_action_chunk[0, 0].cpu())

    print("\nPostprocessed Action[00]:")
    print(robot_action_chunk[0, 0])

    # 比较 PostProcessor 前后数值有没有变化
    diff = (
        model_action_chunk.cpu() - robot_action_chunk
    ).abs().max().item()

    print("\nMax absolute difference:", diff)

    print("\n===== 7. All Postprocessed Actions =====")

    for i in range(robot_action_chunk.shape[1]):
        print(
            f"Action[{i:02d}]: "
            f"{robot_action_chunk[0, i]}"
        )

    print("\nFull Pipeline Test: PASS")


if __name__ == "__main__":
    main()