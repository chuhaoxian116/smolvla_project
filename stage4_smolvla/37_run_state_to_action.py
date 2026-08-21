# 37_run_state_to_action.py

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import DataProcessorPipeline


CHECKPOINT_DIR = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/36_state_to_action_10000"
)

TASK = "Move all six joints to 100."

IMAGE_H = 64
IMAGE_W = 64

JOINT_DIM = 6
TARGET = 100.0

MAX_STEPS = 20
TARGET_TOLERANCE = 2.0

DEVICE = "cuda"


def setup_plot():
    plt.ion()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    lines = []

    for joint in range(JOINT_DIM):
        line, = ax.plot(
            [],
            [],
            marker="o",
            label=f"Joint {joint + 1}",
        )

        lines.append(line)

    ax.axhline(
        y=TARGET,
        linestyle="--",
        label="Target = 100",
    )

    ax.set_title(
        "SmolVLA State -> Action -> State"
    )

    ax.set_xlabel(
        "Inference Step"
    )

    ax.set_ylabel(
        "Joint Value"
    )

    ax.set_xlim(
        0,
        MAX_STEPS,
    )

    ax.set_ylim(
        -5,
        110,
    )

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    plt.show(
        block=False
    )

    return fig, ax, lines


def update_plot(
    fig,
    ax,
    lines,
    state_history,
):
    steps = list(
        range(len(state_history))
    )

    for joint in range(JOINT_DIM):

        values = [
            state[joint]
            for state in state_history
        ]

        lines[joint].set_data(
            steps,
            values,
        )

    ax.set_xlim(
        0,
        max(
            MAX_STEPS,
            len(state_history),
        ),
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.01)


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Load Model
    # =========================================================================

    print("========== 1. Load Model ==========")

    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_DIR,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    preprocessor = (
        DataProcessorPipeline.from_pretrained(
            CHECKPOINT_DIR,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
    )

    postprocessor = (
        DataProcessorPipeline.from_pretrained(
            CHECKPOINT_DIR,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
    )

    print(f"Checkpoint : {CHECKPOINT_DIR}")
    print(f"Chunk size : {policy.config.chunk_size}")
    print(f"Action dim : {policy.config.action_feature.shape[0]}")

    # =========================================================================
    # 2. Initial State
    # =========================================================================

    print("\n========== 2. Initial State ==========")

    state = torch.zeros(
        JOINT_DIM,
        dtype=torch.float32,
    )

    image = torch.zeros(
        (3, IMAGE_H, IMAGE_W),
        dtype=torch.float32,
    )

    print(
        f"State  : {state.tolist()}"
    )

    print(
        f"Target : {[TARGET] * JOINT_DIM}"
    )

    # =========================================================================
    # 3. Fixed Noise
    # =========================================================================

    device = next(
        policy.parameters()
    ).device

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    noise = policy.model.sample_noise(
        noise_shape,
        device,
    )

    # =========================================================================
    # 4. Plot
    # =========================================================================

    print("\n========== 3. Open Curve Window ==========")

    fig, ax, lines = setup_plot()

    state_history = [
        state.tolist()
    ]

    update_plot(
        fig,
        ax,
        lines,
        state_history,
    )

    # =========================================================================
    # 5. State -> Action -> State
    # =========================================================================

    print("\n========== 4. Run SmolVLA ==========")

    for step in range(MAX_STEPS):

        observation = {
            "observation.state":
                state.clone(),

            "observation.images.camera":
                image.clone(),

            "task":
                TASK,
        }

        processed_obs = preprocessor(
            observation
        )

        if hasattr(policy, "reset"):
            policy.reset()

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):

                raw_chunk = (
                    policy.predict_action_chunk(
                        processed_obs,
                        noise=noise.clone(),
                    )
                )

        action_chunk = (
            postprocessor(
                {
                    "action": raw_chunk,
                }
            )["action"]
            .detach()
            .float()
            .cpu()
        )

        # chunk_size = 1
        action = action_chunk[
            0,
            0,
        ]

        error = (
            TARGET - action
        ).abs()

        print()

        print(
            f"Step {step:02d}"
        )

        print(
            "State  :",
            [
                round(value, 3)
                for value
                in state.tolist()
            ],
        )

        print(
            "Action :",
            [
                round(value, 3)
                for value
                in action.tolist()
            ],
        )

        print(
            f"Max target error : "
            f"{error.max().item():.3f}"
        )

        # 仿真：机器人完全执行 Action。
        state = action.clone()

        state_history.append(
            state.tolist()
        )

        update_plot(
            fig,
            ax,
            lines,
            state_history,
        )

        if (
            error.max().item()
            <= TARGET_TOLERANCE
        ):
            print(
                "\nTarget reached."
            )
            break

    # =========================================================================
    # 6. Result
    # =========================================================================

    print("\n========== 5. Result ==========")

    print(
        "Final State:",
        [
            round(value, 3)
            for value
            in state.tolist()
        ],
    )

    print(
        "Target     :",
        [TARGET] * JOINT_DIM,
    )

    print("\n========== Main Flow ==========")

    print(
        """
State[0]
   │
   ▼
SmolVLA
   │
   ▼
Action[0]
   │
   ▼
State = Action
   │
   ▼
SmolVLA
   │
   ▼
Action[1]
   │
   ▼
State = Action
   │
   ▼
  ...
   │
   ▼
Target = 100
"""
    )

    print("========== DONE ==========")

    # 保持窗口打开。
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()