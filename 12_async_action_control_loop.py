import queue
import statistics
import threading
import time

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"

ACTION_HZ = 20.0
ACTION_PERIOD = 1.0 / ACTION_HZ

# Each chunk contains 50 actions.
# This test executes only the first 10 actions.
EXECUTE_STEPS = 10

# Start planning the next chunk after 5 actions have been sent.
REPLAN_TRIGGER_STEP = 5

# Number of action chunks to execute.
CONTROL_CYCLES = 5

WARMUP_RUNS = 3


class MockRobotAdapter:
    def __init__(self):
        self._lock = threading.Lock()

        self._state = torch.tensor(
            [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
            dtype=torch.float32,
        )

    def get_state(self):
        with self._lock:
            return self._state.clone()

    def send_action(self, action):
        if action.ndim != 1:
            raise ValueError(
                f"Expected 1-D action, got {tuple(action.shape)}"
            )

        if action.shape[0] != 6:
            raise ValueError(
                f"Expected action dim 6, got {action.shape[0]}"
            )

        # Mock only:
        # treat the received action as the new robot state.
        with self._lock:
            self._state = action.clone()


def build_observation(state):
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


def run_inference(
    policy,
    preprocessor,
    postprocessor,
    observation,
):
    preprocess_start = time.perf_counter()

    processed = preprocessor(observation)

    torch.cuda.synchronize()

    preprocess_end = time.perf_counter()

    torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.inference_mode():
        model_action_chunk = policy.predict_action_chunk(
            processed
        )

    torch.cuda.synchronize()

    inference_end = time.perf_counter()

    postprocess_start = time.perf_counter()

    postprocessed_action_chunk = postprocessor(
        model_action_chunk
    )

    postprocess_end = time.perf_counter()

    action_chunk = postprocessed_action_chunk[0]

    return {
        "action_chunk": action_chunk,
        "preprocess_ms": (
            preprocess_end - preprocess_start
        ) * 1000.0,
        "inference_ms": (
            inference_end - inference_start
        ) * 1000.0,
        "postprocess_ms": (
            postprocess_end - postprocess_start
        ) * 1000.0,
        "inference_start": inference_start,
        "inference_end": inference_end,
    }


def planner_worker(
    policy,
    preprocessor,
    postprocessor,
    request_queue,
    result_queue,
):
    while True:
        request = request_queue.get()

        if request is None:
            request_queue.task_done()
            break

        plan_id = request["plan_id"]
        observation = request["observation"]
        request_time = request["request_time"]

        result = run_inference(
            policy,
            preprocessor,
            postprocessor,
            observation,
        )

        result["plan_id"] = plan_id
        result["request_time"] = request_time
        result["ready_time"] = time.perf_counter()

        result_queue.put(result)

        request_queue.task_done()


def sleep_until(target_time):
    now = time.perf_counter()

    if now < target_time:
        time.sleep(target_time - now)


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

    print("\n===== 2. Warmup =====")

    warmup_observation = build_observation(
        robot.get_state()
    )

    warmup_processed = preprocessor(
        warmup_observation
    )

    with torch.inference_mode():
        for i in range(WARMUP_RUNS):
            _ = policy.predict_action_chunk(
                warmup_processed
            )

            torch.cuda.synchronize()

            print(
                f"Warmup [{i + 1}/{WARMUP_RUNS}] PASS"
            )

    print("\n===== 3. Initial Plan =====")

    initial_observation = build_observation(
        robot.get_state()
    )

    initial_result = run_inference(
        policy,
        preprocessor,
        postprocessor,
        initial_observation,
    )

    current_chunk = initial_result["action_chunk"]

    print(
        f"Initial inference : "
        f"{initial_result['inference_ms']:.2f} ms"
    )

    print(
        f"Initial chunk     : "
        f"{tuple(current_chunk.shape)}"
    )

    request_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    planner_thread = threading.Thread(
        target=planner_worker,
        args=(
            policy,
            preprocessor,
            postprocessor,
            request_queue,
            result_queue,
        ),
        daemon=True,
    )

    planner_thread.start()

    print("\n===== 4. Async Control Loop =====")

    print(f"Action frequency     : {ACTION_HZ:.1f} Hz")
    print(f"Action period        : {ACTION_PERIOD * 1000:.1f} ms")
    print(f"Execute steps        : {EXECUTE_STEPS}")
    print(f"Replan trigger step  : {REPLAN_TRIGGER_STEP}")
    print(f"Control cycles       : {CONTROL_CYCLES}")

    inference_times = []
    boundary_wait_times = []

    last_inference_end = initial_result[
        "inference_end"
    ]

    program_start = time.perf_counter()

    for cycle in range(CONTROL_CYCLES):
        print(
            f"\n========== Control Cycle "
            f"[{cycle + 1}/{CONTROL_CYCLES}] =========="
        )

        cycle_start = time.perf_counter()

        print("Start State:")
        print(robot.get_state())

        next_plan_requested = False

        for step in range(EXECUTE_STEPS):
            target_time = (
                cycle_start
                + step * ACTION_PERIOD
            )

            sleep_until(target_time)

            actual_time = time.perf_counter()

            action = current_chunk[step]

            robot.send_action(action)

            elapsed_ms = (
                actual_time - cycle_start
            ) * 1000.0

            print(
                f"Action[{step:02d}] "
                f"t={elapsed_ms:7.2f} ms -> "
                f"{action}"
            )

            # Request the next chunk while the robot
            # continues executing the current chunk.
            if (
                not next_plan_requested
                and cycle < CONTROL_CYCLES - 1
                and step + 1 == REPLAN_TRIGGER_STEP
            ):
                state_snapshot = robot.get_state()

                observation = build_observation(
                    state_snapshot
                )

                request_time = time.perf_counter()

                request_queue.put(
                    {
                        "plan_id": cycle + 2,
                        "observation": observation,
                        "request_time": request_time,
                    }
                )

                next_plan_requested = True

                request_elapsed_ms = (
                    request_time - cycle_start
                ) * 1000.0

                print(
                    f"\n[Planner] Request Plan "
                    f"#{cycle + 2} at "
                    f"t={request_elapsed_ms:.2f} ms"
                )

                print(
                    "[Planner] State snapshot:"
                )
                print(state_snapshot)
                print()

        # Wait until the full 10-action time window
        # has completed.
        boundary_time = (
            cycle_start
            + EXECUTE_STEPS * ACTION_PERIOD
        )

        sleep_until(boundary_time)

        actual_boundary = time.perf_counter()

        print(
            f"\nChunk boundary at "
            f"{(actual_boundary - cycle_start) * 1000.0:.2f} ms"
        )

        if cycle < CONTROL_CYCLES - 1:
            wait_start = time.perf_counter()

            result_was_ready = not result_queue.empty()

            next_result = result_queue.get()

            wait_end = time.perf_counter()

            boundary_wait_ms = (
                wait_end - wait_start
            ) * 1000.0

            boundary_wait_times.append(
                boundary_wait_ms
            )

            inference_ms = next_result[
                "inference_ms"
            ]

            inference_times.append(
                inference_ms
            )

            planner_delay_ms = (
                next_result["inference_start"]
                - next_result["request_time"]
            ) * 1000.0

            plan_ready_after_request_ms = (
                next_result["ready_time"]
                - next_result["request_time"]
            ) * 1000.0

            idle_before_inference_ms = (
                next_result["inference_start"]
                - last_inference_end
            ) * 1000.0

            last_inference_end = next_result[
                "inference_end"
            ]

            print("\n----- Planner Result -----")

            print(
                f"Plan ID                  : "
                f"{next_result['plan_id']}"
            )

            print(
                f"PreProcess               : "
                f"{next_result['preprocess_ms']:.2f} ms"
            )

            print(
                f"Inference                : "
                f"{inference_ms:.2f} ms"
            )

            print(
                f"PostProcess              : "
                f"{next_result['postprocess_ms']:.2f} ms"
            )

            print(
                f"Planner queue delay      : "
                f"{planner_delay_ms:.2f} ms"
            )

            print(
                f"Idle before inference    : "
                f"{idle_before_inference_ms:.2f} ms"
            )

            print(
                f"Plan ready after request : "
                f"{plan_ready_after_request_ms:.2f} ms"
            )

            print(
                f"Ready before boundary    : "
                f"{result_was_ready}"
            )

            print(
                f"Robot boundary wait      : "
                f"{boundary_wait_ms:.2f} ms"
            )

            current_chunk = next_result[
                "action_chunk"
            ]

            result_queue.task_done()

            print(
                "\n-> Switch to new Action Chunk"
            )

        print("\nEnd State:")
        print(robot.get_state())

    total_time = (
        time.perf_counter() - program_start
    )

    print("\n===== 5. Stop Planner =====")

    request_queue.put(None)

    planner_thread.join()

    print("Planner thread stopped.")

    print("\n===== 6. Summary =====")

    if inference_times:
        print(
            f"Average async inference : "
            f"{statistics.mean(inference_times):.2f} ms"
        )

        print(
            f"Min async inference     : "
            f"{min(inference_times):.2f} ms"
        )

        print(
            f"Max async inference     : "
            f"{max(inference_times):.2f} ms"
        )

    if boundary_wait_times:
        print(
            f"Average boundary wait   : "
            f"{statistics.mean(boundary_wait_times):.2f} ms"
        )

        print(
            f"Max boundary wait       : "
            f"{max(boundary_wait_times):.2f} ms"
        )

    print(
        f"Total control time      : "
        f"{total_time:.2f} s"
    )

    print("\nFinal Robot State:")
    print(robot.get_state())

    print(
        "\nAsync Action Control Loop Test: PASS"
    )


if __name__ == "__main__":
    main()