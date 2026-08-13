import statistics
import time

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

WARMUP_RUNS = 3
TEST_RUNS = 5

# Idle time between two inference calls.
IDLE_GAPS_MS = [
    0,
    50,
    100,
    200,
    300,
    500,
    1000,
]


def build_observation():
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


def measure_inference(policy, processed):
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(processed)

    torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0

    return latency_ms, action_chunk


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

    observation = build_observation()
    processed = preprocessor(observation)

    print("\n===== 2. Warmup =====")

    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            _ = policy.predict_action_chunk(processed)
            torch.cuda.synchronize()

            print(
                f"Warmup [{i + 1}/{WARMUP_RUNS}] PASS"
            )

    print("\n===== 3. Idle Gap Test =====")

    results = {}

    for gap_ms in IDLE_GAPS_MS:
        gap_s = gap_ms / 1000.0

        print(
            f"\n----- Idle Gap: {gap_ms} ms -----"
        )

        # One inference before each group keeps the starting
        # condition reasonably consistent.
        with torch.inference_mode():
            _ = policy.predict_action_chunk(processed)

        torch.cuda.synchronize()

        latencies = []

        for i in range(TEST_RUNS):
            if gap_s > 0:
                time.sleep(gap_s)

            latency_ms, _ = measure_inference(
                policy,
                processed,
            )

            latencies.append(latency_ms)

            print(
                f"Run [{i + 1:02d}/{TEST_RUNS}] "
                f"Gap={gap_ms:4d} ms  "
                f"Inference={latency_ms:.2f} ms"
            )

        avg_ms = statistics.mean(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)

        results[gap_ms] = {
            "avg": avg_ms,
            "min": min_ms,
            "max": max_ms,
        }

        print(
            f"Average : {avg_ms:.2f} ms"
        )
        print(
            f"Minimum : {min_ms:.2f} ms"
        )
        print(
            f"Maximum : {max_ms:.2f} ms"
        )

    print("\n===== 4. Final Comparison =====")

    baseline_ms = results[0]["avg"]

    print(
        f"{'Idle Gap':>10}  "
        f"{'Avg Inference':>15}  "
        f"{'Delta':>10}  "
        f"{'Rate':>10}"
    )

    print("-" * 55)

    for gap_ms in IDLE_GAPS_MS:
        avg_ms = results[gap_ms]["avg"]
        delta_ms = avg_ms - baseline_ms
        rate_hz = 1000.0 / avg_ms

        print(
            f"{gap_ms:>7} ms  "
            f"{avg_ms:>12.2f} ms  "
            f"{delta_ms:>+8.2f} ms  "
            f"{rate_hz:>8.2f} Hz"
        )

    print("\n===== 5. Result =====")

    gap_500_ms = results[500]["avg"]

    print(
        f"Continuous baseline : "
        f"{baseline_ms:.2f} ms"
    )

    print(
        f"After 500 ms idle   : "
        f"{gap_500_ms:.2f} ms"
    )

    print(
        f"500 ms idle penalty : "
        f"{gap_500_ms - baseline_ms:+.2f} ms"
    )

    print(
        "\nInference Idle Gap Test: PASS"
    )


if __name__ == "__main__":
    main()
