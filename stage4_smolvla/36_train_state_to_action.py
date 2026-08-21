# 36_train_state_to_action.py

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import FeatureType
from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


REPO_ID = "local/state_to_action_10000"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/state_to_action_10000"
)

CHECKPOINT_DIR = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/36_state_to_action_10000"
)

MODEL_ID = "lerobot/smolvla_base"

TASK = "Move all six joints to 100."

JOINT_DIM = 6

STATE_MIN = 0.0
STATE_MAX = 100.0

ACTION_STEP = 10.0
EXECUTION_NOISE = 0.2

TOTAL_SAMPLES = 10000

RANDOM_RATIO = 0.70
ROLLOUT_RATIO = 0.30

IMAGE_H = 64
IMAGE_W = 64

FPS = 10

CHUNK_SIZE = 1
BATCH_SIZE = 32
NUM_EPOCHS = 5

LEARNING_RATE = 1e-4
DEVICE = "cuda"

LOCAL_FILES_ONLY = True

RESET_DATASET = True
RESET_CHECKPOINT = True

SEED = 42


def expert_action(state):
    return np.clip(
        state + ACTION_STEP,
        STATE_MIN,
        STATE_MAX,
    ).astype(np.float32)


def create_dataset():
    random.seed(SEED)
    np.random.seed(SEED)

    if DATASET_ROOT.exists() and RESET_DATASET:
        shutil.rmtree(DATASET_ROOT)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        },

        "observation.images.camera": {
            "dtype": "image",
            "shape": (IMAGE_H, IMAGE_W, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        },

        "action": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [
                "target_joint_1",
                "target_joint_2",
                "target_joint_3",
                "target_joint_4",
                "target_joint_5",
                "target_joint_6",
            ],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        fps=FPS,
        robot_type="mock_state_to_action",
        features=features,
        use_videos=False,
    )

    image = np.zeros(
        (IMAGE_H, IMAGE_W, 3),
        dtype=np.uint8,
    )

    random_samples = int(
        TOTAL_SAMPLES * RANDOM_RATIO
    )

    rollout_samples = (
        TOTAL_SAMPLES - random_samples
    )

    print("========== Create Dataset ==========")

    print(f"Total Samples  : {TOTAL_SAMPLES}")
    print(f"Random Samples : {random_samples}")
    print(f"Rollout Samples: {rollout_samples}")
    print(
        f"Execution Noise: "
        f"[-{EXECUTION_NOISE}, +{EXECUTION_NOISE}]"
    )

    # =========================================================================
    # 1. Random State
    # =========================================================================

    print("\n========== 1. Random State Samples ==========")

    for index in range(random_samples):

        state = np.random.uniform(
            STATE_MIN,
            STATE_MAX,
            size=JOINT_DIM,
        ).astype(np.float32)

        action = expert_action(
            state
        )

        dataset.add_frame(
            {
                "observation.state": state,
                "observation.images.camera": image.copy(),
                "action": action,
                "task": TASK,
            }
        )

        if (
            index < 5
            or (index + 1) % 1000 == 0
        ):
            print(
                f"[Random {index + 1:04d}] "
                f"State={np.round(state, 2)} "
                f"-> "
                f"Action={np.round(action, 2)}"
            )

    # =========================================================================
    # 2. Noisy Rollout
    # =========================================================================

    print("\n========== 2. Noisy Rollout Samples ==========")

    state = np.zeros(
        JOINT_DIM,
        dtype=np.float32,
    )

    for index in range(rollout_samples):

        action = expert_action(
            state
        )

        dataset.add_frame(
            {
                "observation.state": state.copy(),
                "observation.images.camera": image.copy(),
                "action": action.copy(),
                "task": TASK,
            }
        )

        noise = np.random.uniform(
            -EXECUTION_NOISE,
            EXECUTION_NOISE,
            size=JOINT_DIM,
        ).astype(np.float32)

        next_state = (
            action + noise
        )

        next_state = np.clip(
            next_state,
            STATE_MIN,
            STATE_MAX,
        ).astype(np.float32)

        if (
            index < 10
            or (index + 1) % 500 == 0
        ):
            print(
                f"[Rollout {index + 1:04d}] "
                f"State={np.round(state, 2)} "
                f"-> "
                f"Action={np.round(action, 2)} "
                f"-> "
                f"Next={np.round(next_state, 2)}"
            )

        state = next_state

        # 到达目标以后重新从 0 开始下一条轨迹。
        if np.all(
            state >= STATE_MAX - EXECUTION_NOISE
        ):
            state = np.zeros(
                JOINT_DIM,
                dtype=np.float32,
            )

    dataset.save_episode()
    dataset.finalize()

    print(
        f"\nDataset saved: "
        f"{DATASET_ROOT}"
    )


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # =========================================================================
    # 1. Dataset
    # =========================================================================

    print("\n========== 1. Dataset ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    policy_features = (
        dataset_to_policy_features(
            metadata.features
        )
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

    # =========================================================================
    # 2. Config
    # =========================================================================

    print("\n========== 2. Config ==========")

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        device=DEVICE,
        load_vlm_weights=False,
    )

    print(
        f"State dim  : "
        f"{cfg.robot_state_feature.shape[0]}"
    )

    print(
        f"Action dim : "
        f"{cfg.action_feature.shape[0]}"
    )

    print(
        f"Chunk size : "
        f"{cfg.chunk_size}"
    )

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index
            in cfg.action_delta_indices
        ]
    }

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Samples         : {len(dataset)}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Batches / epoch : {len(dataloader)}")
    print(f"Epochs          : {NUM_EPOCHS}")

    print(
        f"Total steps     : "
        f"{len(dataloader) * NUM_EPOCHS}"
    )

    # =========================================================================
    # 3. Processor
    # =========================================================================

    print("\n========== 3. Processor ==========")

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=cfg,
            dataset_stats=metadata.stats,
        )
    )

    print("PreProcessor  : PASS")
    print("PostProcessor : PASS")

    # =========================================================================
    # 4. Model
    # =========================================================================

    print("\n========== 4. Load SmolVLA ==========")

    policy = SmolVLAPolicy.from_pretrained(
        MODEL_ID,
        config=cfg,
        local_files_only=LOCAL_FILES_ONLY,
        strict=False,
    )

    policy.train()

    trainable_params = [
        parameter
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
    )

    print(
        f"Device : "
        f"{next(policy.parameters()).device}"
    )

    print("Mode   : train")

    print(
        f"Trainable parameters : "
        f"{sum(p.numel() for p in trainable_params):,}"
    )

    # =========================================================================
    # 5. Training
    # =========================================================================

    print("\n========== 5. Training ==========")

    epoch_losses = []
    global_step = 0

    for epoch in range(NUM_EPOCHS):

        loss_sum = 0.0
        steps = 0

        for raw_batch in dataloader:

            for camera_key in metadata.camera_keys:

                if (
                    camera_key in raw_batch
                    and isinstance(
                        raw_batch[camera_key],
                        torch.Tensor,
                    )
                    and raw_batch[camera_key].dtype
                    == torch.uint8
                ):
                    raw_batch[camera_key] = (
                        raw_batch[camera_key]
                        .float()
                        / 255.0
                    )

            batch = preprocessor(
                raw_batch
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(DEVICE == "cuda"),
            ):
                loss, _ = policy.forward(
                    batch
                )

            loss.backward()

            optimizer.step()

            loss_value = (
                loss.detach().item()
            )

            loss_sum += loss_value

            steps += 1
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"Epoch "
                    f"{epoch + 1:02d}/{NUM_EPOCHS} | "
                    f"Step {global_step:05d} | "
                    f"Loss={loss_value:.6f}"
                )

        average_loss = (
            loss_sum / steps
        )

        epoch_losses.append(
            average_loss
        )

        print(
            f">>> Epoch "
            f"{epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Average Loss="
            f"{average_loss:.6f}"
        )

    # =========================================================================
    # 6. Save
    # =========================================================================

    print("\n========== 6. Save Checkpoint ==========")

    if (
        CHECKPOINT_DIR.exists()
        and RESET_CHECKPOINT
    ):
        shutil.rmtree(
            CHECKPOINT_DIR
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(
        CHECKPOINT_DIR
    )

    preprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    postprocessor.save_pretrained(
        CHECKPOINT_DIR
    )

    training_info = {
        "dataset": REPO_ID,
        "samples": TOTAL_SAMPLES,
        "random_ratio": RANDOM_RATIO,
        "rollout_ratio": ROLLOUT_RATIO,
        "state_dim": JOINT_DIM,
        "action_dim": JOINT_DIM,
        "state_range": [
            STATE_MIN,
            STATE_MAX,
        ],
        "action_step": ACTION_STEP,
        "execution_noise": EXECUTION_NOISE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_steps": global_step,
        "learning_rate": LEARNING_RATE,
        "epoch_losses": epoch_losses,
    }

    with open(
        CHECKPOINT_DIR / "training_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_info,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Checkpoint : "
        f"{CHECKPOINT_DIR}"
    )

    print(
        f"Training Steps : "
        f"{global_step}"
    )

    print(
        f"Final Loss     : "
        f"{epoch_losses[-1]:.6f}"
    )

    # =========================================================================
    # 7. Main Flow
    # =========================================================================

    print("\n========== Main Flow ==========")

    print(
        """
             10000 Samples

        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
  70% Random State     30% Noisy Rollout
        │                     │
        │              State(t)
        │                 │
        │                 ▼
        │              Expert
        │                 │
        │                 ▼
        │              Action(t)
        │                 │
        │            + [-0.2,0.2]
        │                 │
        │                 ▼
        │            State(t+1)
        │
        └──────────┬──────────┘
                   ▼
                State[6]
                   │
                   ▼
                SmolVLA
                   │
                   ▼
          Predicted Action[6]
                   │
                   ▼
            Expert Action[6]
                   │
                   ▼
                  Loss
                   │
                   ▼
                Backward
                   │
                   ▼
               Optimizer
"""
    )

    print("========== DONE ==========")


def main():
    create_dataset()
    train()


if __name__ == "__main__":
    main()