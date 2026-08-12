import time

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

# 预热次数：第一次 CUDA 推理通常比较慢
WARMUP_RUNS = 3

# 正式测试次数
TEST_RUNS = 10


def build_fake_observation():
    """构造假的 State + Camera + Task。"""

    return {
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


def main():
    print("===== 1. Load Model =====")

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    print("\n===== 2. Build Observation =====")

    observation = build_fake_observation()

    # 提前执行 PreProcessor。
    # 本测试只测 SmolVLA 模型推理，不统计数据预处理时间。
    processed = preprocessor(observation)

    print("State :", processed["observation.state"].shape)
    print("Camera:", processed["observation.images.camera1"].shape)
    print("Tokens:", processed["observation.language.tokens"].shape)

    print("\n===== 3. Warmup =====")

    # CUDA 第一次运行通常包含初始化开销，
    # 所以先运行几次，不计入最终结果。
    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            _ = policy.predict_action_chunk(processed)

            # CUDA 默认异步执行，等待 GPU 真正计算完成
            torch.cuda.synchronize()

            print(f"Warmup [{i + 1}/{WARMUP_RUNS}] PASS")

    print("\n===== 4. Measure Inference Latency =====")

    latencies_ms = []

    with torch.inference_mode():
        for i in range(TEST_RUNS):

            # 确保之前的 CUDA 任务已经完成
            torch.cuda.synchronize()

            start_time = time.perf_counter()

            # 一次完整 SmolVLA 推理：
            # Observation -> Action Chunk [1, 50, 6]
            action_chunk = policy.predict_action_chunk(processed)

            # 等待 GPU 真正完成推理后再停止计时
            torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latencies_ms.append(latency_ms)

            print(
                f"Inference [{i + 1:02d}/{TEST_RUNS}]: "
                f"{latency_ms:.2f} ms"
            )

    print("\n===== 5. Result =====")

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    # 根据平均推理时间计算理论最大推理频率
    inference_hz = 1000.0 / avg_ms

    print(f"Average : {avg_ms:.2f} ms")
    print(f"Minimum : {min_ms:.2f} ms")
    print(f"Maximum : {max_ms:.2f} ms")
    print(f"Theoretical inference rate: {inference_hz:.2f} Hz")

    print("\nAction Chunk:")
    print("  shape :", tuple(action_chunk.shape))
    print("  device:", action_chunk.device)

    print("\nInference Latency Test: PASS")


if __name__ == "__main__":
    main()