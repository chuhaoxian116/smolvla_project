import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"


def main():
    print("===== 1. Load Model =====")

    # 加载 SmolVLA 模型和预训练权重
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)

    # 创建输入前处理器和输出后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    print("\n===== 2. Build Fake Observation =====")

    # 构造假的 Observation：
    # 6维 State + 三张 RGB 图片 + 任务描述
    fake_observation = {
        "observation.state": torch.tensor(
            [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
            dtype=torch.float32,
        ),

        # shape = [C, H, W] = [3, 256, 256]
        # 全 0 表示黑色假图片
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

    # 原始 Observation → SmolVLA 可直接使用的数据
    # 这里会完成 Batch、Tokenizer、CUDA、Normalization 等处理
    processed = preprocessor(fake_observation)

    print("State :", processed["observation.state"].shape)
    print("Camera:", processed["observation.images.camera1"].shape)
    print("Tokens:", processed["observation.language.tokens"].shape)

    print("\n===== 4. SmolVLA Predict Action Chunk =====")

    # 只做推理，不计算训练梯度
    with torch.no_grad():
        # 输入 Observation，生成整段 Action Chunk
        action_chunk = policy.predict_action_chunk(processed)

    print("\n===== 5. Action Chunk Result =====")

    # 预期 shape = [Batch, Chunk, ActionDim] = [1, 50, 6]
    print("Shape :", tuple(action_chunk.shape))
    print("dtype :", action_chunk.dtype)
    print("device:", action_chunk.device)

    # 查看第 1、2、最后一个 Action
    # print("\nFirst action:")
    # print(action_chunk[0, 0].cpu())

    # print("\nSecond action:")
    # print(action_chunk[0, 1].cpu())

    # print("\nLast action:")
    # print(action_chunk[0, -1].cpu())

    print("\n===== All Actions =====")

    for i in range(action_chunk.shape[1]):
        action = action_chunk[0, i].cpu()

        print(f"Action[{i:02d}]: {action}")

    # 查看整个 Action Chunk 的数值范围
    print("\nMin:", action_chunk.min().item())
    print("Max:", action_chunk.max().item())

    print("\nSmolVLA Action Chunk Test: PASS")


if __name__ == "__main__":
    main()