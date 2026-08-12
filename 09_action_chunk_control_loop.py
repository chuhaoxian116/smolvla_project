import time
import statistics

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

# 模型输出 50 个 Action，只执行前 10 个
EXECUTE_STEPS = 10

# Action 执行频率
ACTION_HZ = 20.0
ACTION_PERIOD = 1.0 / ACTION_HZ

# 重新规划次数
REPLAN_CYCLES = 10

# CUDA / 模型预热次数
WARMUP_RUNS = 3


class MockRobotAdapter:
    """模拟机器人执行层。"""

    def __init__(self):
        self.state = torch.tensor(
            [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
            dtype=torch.float32,
        )

    def get_state(self):
        """读取当前机器人状态。"""
        return self.state.clone()

    def send_action(self, action):
        """
        模拟执行一个 Action。

        当前只是软件闭环测试：
        暂时直接把 Action 当成新的 State。
        """
        if action.shape != (6,):
            raise ValueError(
                f"Expected action shape (6,), got {tuple(action.shape)}"
            )

        self.state = action.clone()


def build_observation(current_state):
    """根据当前机器人 State 构造 Observation。"""

    return {
        "observation.state": current_state,

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
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("===== 1. Load Model =====")

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    robot = MockRobotAdapter()

    # ============================================================
    # Warmup
    # ============================================================

    print("\n===== 2. Warmup =====")

    warmup_state = robot.get_state()
    warmup_observation = build_observation(warmup_state)
    warmup_processed = preprocessor(warmup_observation)

    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            _ = policy.predict_action_chunk(warmup_processed)

            # 等待 CUDA 真正完成计算
            torch.cuda.synchronize()

            print(f"Warmup [{i + 1}/{WARMUP_RUNS}] PASS")

    # ============================================================
    # Control Loop
    # ============================================================

    print("\n===== 3. Start Control Loop =====")

    print(f"Action frequency : {ACTION_HZ:.1f} Hz")
    print(f"Action period    : {ACTION_PERIOD * 1000:.1f} ms")
    print(f"Execute steps    : {EXECUTE_STEPS}")
    print(f"Replan cycles    : {REPLAN_CYCLES}")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    execute_times = []
    cycle_times = []

    for cycle in range(REPLAN_CYCLES):

        print(
            f"\n========== Replan Cycle "
            f"[{cycle + 1}/{REPLAN_CYCLES}] =========="
        )

        cycle_start = time.perf_counter()

        # --------------------------------------------------------
        # 1. 获取当前机器人 State
        # --------------------------------------------------------

        current_state = robot.get_state()

        print("Current State:")
        print(current_state)

        # --------------------------------------------------------
        # 2. 构造 Observation
        # --------------------------------------------------------

        observation = build_observation(current_state)

        # --------------------------------------------------------
        # 3. PreProcess
        # --------------------------------------------------------

        preprocess_start = time.perf_counter()

        processed = preprocessor(observation)

        # Processor 中有 CUDA 数据搬运，因此同步后再计时
        torch.cuda.synchronize()

        preprocess_end = time.perf_counter()

        preprocess_ms = (
            preprocess_end - preprocess_start
        ) * 1000.0

        # --------------------------------------------------------
        # 4. SmolVLA 推理
        # --------------------------------------------------------

        torch.cuda.synchronize()
        inference_start = time.perf_counter()

        with torch.inference_mode():
            model_action_chunk = policy.predict_action_chunk(
                processed
            )

        torch.cuda.synchronize()
        inference_end = time.perf_counter()

        inference_ms = (
            inference_end - inference_start
        ) * 1000.0

        # --------------------------------------------------------
        # 5. PostProcess
        # --------------------------------------------------------

        postprocess_start = time.perf_counter()

        postprocessed_action_chunk = postprocessor(
            model_action_chunk
        )

        postprocess_end = time.perf_counter()

        postprocess_ms = (
            postprocess_end - postprocess_start
        ) * 1000.0

        # [1, 50, 6] -> [50, 6]
        action_chunk = postprocessed_action_chunk[0]

        print(f"\nSmolVLA predicted : {action_chunk.shape[0]} actions")
        print(f"Execute           : first {EXECUTE_STEPS} actions")
        print(
            f"Discard           : "
            f"remaining {action_chunk.shape[0] - EXECUTE_STEPS} actions"
        )

        # --------------------------------------------------------
        # 6. 执行前 10 个 Action
        # --------------------------------------------------------

        execute_start = time.perf_counter()

        for i in range(EXECUTE_STEPS):

            action = action_chunk[i]

            step_start = time.perf_counter()

            robot.send_action(action)

            print(
                f"Action[{i:02d}] -> "
                f"{action}"
            )

            # 保持固定 20Hz：
            # 每 50ms 执行一个 Action
            elapsed = time.perf_counter() - step_start
            sleep_time = ACTION_PERIOD - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        execute_end = time.perf_counter()

        execute_ms = (
            execute_end - execute_start
        ) * 1000.0

        # --------------------------------------------------------
        # 7. 本轮总时间
        # --------------------------------------------------------

        cycle_end = time.perf_counter()

        cycle_ms = (
            cycle_end - cycle_start
        ) * 1000.0

        ai_pipeline_ms = (
            preprocess_ms
            + inference_ms
            + postprocess_ms
        )

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        execute_times.append(execute_ms)
        cycle_times.append(cycle_ms)

        print("\n----- Timing -----")
        print(f"PreProcess       : {preprocess_ms:.2f} ms")
        print(f"Inference        : {inference_ms:.2f} ms")
        print(f"PostProcess      : {postprocess_ms:.2f} ms")
        print(f"AI Pipeline      : {ai_pipeline_ms:.2f} ms")
        print(f"Execute 10 Steps : {execute_ms:.2f} ms")
        print(f"Total Cycle      : {cycle_ms:.2f} ms")

        print("\nNew Mock Robot State:")
        print(robot.get_state())

        print("-> Re-observe and replan")

    # ============================================================
    # Summary
    # ============================================================

    print("\n===== 4. Timing Summary =====")

    avg_preprocess = statistics.mean(preprocess_times)
    avg_inference = statistics.mean(inference_times)
    avg_postprocess = statistics.mean(postprocess_times)
    avg_execute = statistics.mean(execute_times)
    avg_cycle = statistics.mean(cycle_times)

    avg_ai_pipeline = (
        avg_preprocess
        + avg_inference
        + avg_postprocess
    )

    # 当前同步控制模式下的实际重新规划频率
    effective_replan_hz = 1000.0 / avg_cycle

    # 单独看模型能够连续推理的理论频率
    model_inference_hz = 1000.0 / avg_inference

    print(f"Average PreProcess       : {avg_preprocess:.2f} ms")
    print(f"Average Inference        : {avg_inference:.2f} ms")
    print(f"Average PostProcess      : {avg_postprocess:.2f} ms")
    print(f"Average AI Pipeline      : {avg_ai_pipeline:.2f} ms")
    print(f"Average Execute 10 Steps : {avg_execute:.2f} ms")
    print(f"Average Total Cycle      : {avg_cycle:.2f} ms")

    print()
    print(
        f"Model inference rate     : "
        f"{model_inference_hz:.2f} Hz"
    )

    print(
        f"Effective replan rate    : "
        f"{effective_replan_hz:.2f} Hz"
    )

    print("\n===== 5. Control Loop Finished =====")

    print("Final Robot State:")
    print(robot.get_state())

    print("\nAction Chunk Control Loop Test: PASS")


if __name__ == "__main__":
    main()