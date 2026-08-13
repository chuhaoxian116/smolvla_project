import time
import statistics

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

WARMUP_RUNS = 3
TEST_RUNS = 10


def build_observation(state):
    """构造固定 Camera + Task，仅 State 可变化。"""

    return {
        "observation.state": state.clone(),

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


def measure_inference(policy, processed):
    """测量一次 predict_action_chunk() 的纯推理时间。"""

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(processed)

    torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0

    return latency_ms, action_chunk


def print_summary(name, latencies):
    avg_ms = statistics.mean(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    print(f"\n----- {name} Summary -----")
    print(f"Average : {avg_ms:.2f} ms")
    print(f"Minimum : {min_ms:.2f} ms")
    print(f"Maximum : {max_ms:.2f} ms")
    print(f"Rate    : {1000.0 / avg_ms:.2f} Hz")

    return avg_ms


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("===== 1. Load Model =====")

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    initial_state = torch.tensor(
        [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
        dtype=torch.float32,
    )

    # ============================================================
    # Warmup
    # ============================================================

    print("\n===== 2. Warmup =====")

    warmup_observation = build_observation(initial_state)
    warmup_processed = preprocessor(warmup_observation)

    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            _ = policy.predict_action_chunk(warmup_processed)

            torch.cuda.synchronize()

            print(f"Warmup [{i + 1}/{WARMUP_RUNS}] PASS")

    # ============================================================
    # Test A
    # 同一个 processed 连续推理
    # ============================================================

    print("\n===== 3. Test A: Same Processed Input =====")

    observation_a = build_observation(initial_state)
    processed_a = preprocessor(observation_a)

    latencies_a = []

    for i in range(TEST_RUNS):
        latency_ms, _ = measure_inference(
            policy,
            processed_a,
        )

        latencies_a.append(latency_ms)

        print(
            f"A [{i + 1:02d}/{TEST_RUNS}] "
            f"{latency_ms:.2f} ms"
        )

    avg_a = print_summary(
        "Test A",
        latencies_a,
    )

    # ============================================================
    # Test B
    # 每轮重新 PreProcess，但 State 不变
    # ============================================================

    print("\n===== 4. Test B: Reprocess Same State =====")

    latencies_b = []
    preprocess_times_b = []

    for i in range(TEST_RUNS):
        observation = build_observation(initial_state)

        preprocess_start = time.perf_counter()

        processed = preprocessor(observation)

        torch.cuda.synchronize()

        preprocess_end = time.perf_counter()

        preprocess_ms = (
            preprocess_end - preprocess_start
        ) * 1000.0

        latency_ms, _ = measure_inference(
            policy,
            processed,
        )

        preprocess_times_b.append(preprocess_ms)
        latencies_b.append(latency_ms)

        print(
            f"B [{i + 1:02d}/{TEST_RUNS}] "
            f"Pre={preprocess_ms:.2f} ms  "
            f"Inference={latency_ms:.2f} ms"
        )

    avg_b = print_summary(
        "Test B Inference",
        latencies_b,
    )

    print(
        f"Average PreProcess B: "
        f"{statistics.mean(preprocess_times_b):.2f} ms"
    )

    # ============================================================
    # Test C
    # 每轮重新 PreProcess，同时 State 每次变化
    # ============================================================

    print("\n===== 5. Test C: Reprocess Changing State =====")

    latencies_c = []
    preprocess_times_c = []

    current_state = initial_state.clone()

    state_delta = torch.tensor(
        [0.10, 0.05, -0.03, 0.08, -0.04, 0.02],
        dtype=torch.float32,
    )

    for i in range(TEST_RUNS):
        # 这里只是人为改变 State，测试输入变化的影响。
        current_state = current_state + state_delta

        observation = build_observation(current_state)

        preprocess_start = time.perf_counter()

        processed = preprocessor(observation)

        torch.cuda.synchronize()

        preprocess_end = time.perf_counter()

        preprocess_ms = (
            preprocess_end - preprocess_start
        ) * 1000.0

        latency_ms, _ = measure_inference(
            policy,
            processed,
        )

        preprocess_times_c.append(preprocess_ms)
        latencies_c.append(latency_ms)

        print(
            f"C [{i + 1:02d}/{TEST_RUNS}] "
            f"Pre={preprocess_ms:.2f} ms  "
            f"Inference={latency_ms:.2f} ms"
        )

    avg_c = print_summary(
        "Test C Inference",
        latencies_c,
    )

    print(
        f"Average PreProcess C: "
        f"{statistics.mean(preprocess_times_c):.2f} ms"
    )

    # ============================================================
    # Final Comparison
    # ============================================================

    print("\n===== 6. Final Comparison =====")

    print(
        f"A Same processed       : "
        f"{avg_a:.2f} ms"
    )

    print(
        f"B Reprocess same state : "
        f"{avg_b:.2f} ms"
    )

    print(
        f"C Changing state       : "
        f"{avg_c:.2f} ms"
    )

    print("\nDifference:")

    print(
        f"B - A : "
        f"{avg_b - avg_a:+.2f} ms"
    )

    print(
        f"C - B : "
        f"{avg_c - avg_b:+.2f} ms"
    )

    print(
        f"C - A : "
        f"{avg_c - avg_a:+.2f} ms"
    )

    print("\nInference State Change Test: PASS")


if __name__ == "__main__":
    main()