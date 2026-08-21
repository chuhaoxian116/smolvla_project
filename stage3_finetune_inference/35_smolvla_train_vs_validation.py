from pathlib import Path

import torch

from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import DataProcessorPipeline


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

CHECKPOINT_DIR = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "checkpoints/34_train_ep01"
)

TRAIN_EPISODES = [0, 1]
VALIDATION_EPISODES = [2]

DEVICE = "cuda"


def evaluate_episode(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    metadata,
    episode_index,
    eval_noise,
):
    episode = metadata.episodes[episode_index]

    start = int(
        episode["dataset_from_index"]
    )
    end = int(
        episode["dataset_to_index"]
    )

    total_abs_error = 0.0
    total_valid_values = 0

    for sample_index in range(start, end):

        sample = dataset[sample_index]

        state = sample[
            "observation.state"
        ].clone()

        image = sample[
            "observation.images.camera"
        ].clone()

        task = sample["task"]

        expert_chunk = (
            sample["action"]
            .detach()
            .float()
            .cpu()
            .unsqueeze(0)
        )

        action_is_pad = sample.get(
            "action_is_pad"
        )

        if action_is_pad is None:
            valid_mask = torch.ones(
                expert_chunk.shape[1],
                dtype=torch.bool,
            )
        else:
            valid_mask = (
                ~action_is_pad
                .detach()
                .cpu()
                .bool()
            )

        if image.dtype == torch.uint8:
            image = (
                image.to(torch.float32)
                / 255.0
            )

        # 只把 Observation 给模型。
        observation = {
            "observation.state": state,
            "observation.images.camera": image,
            "task": task,
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
                        noise=eval_noise.clone(),
                    )
                )

        predicted_chunk = (
            postprocessor(
                {
                    "action": raw_chunk,
                }
            )["action"]
            .detach()
            .float()
            .cpu()
        )

        abs_error = (
            predicted_chunk
            - expert_chunk
        ).abs()

        mask = (
            valid_mask
            .view(1, -1, 1)
            .expand_as(abs_error)
        )

        total_abs_error += (
            abs_error[mask]
            .sum()
            .item()
        )

        total_valid_values += (
            mask.sum().item()
        )

    mae = (
        total_abs_error
        / total_valid_values
    )

    return {
        "episode": episode_index,
        "samples": end - start,
        "abs_error_sum": total_abs_error,
        "valid_values": total_valid_values,
        "mae": mae,
    }


def main():
    torch.manual_seed(42)

    # =========================================================================
    # 1. Dataset
    # =========================================================================
    print("========== 1. Dataset ==========")

    metadata = LeRobotDatasetMetadata(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(f"Episodes : {metadata.total_episodes}")
    print(f"Frames   : {metadata.total_frames}")
    print(f"FPS      : {metadata.fps}")

    # =========================================================================
    # 2. Load Checkpoint
    # =========================================================================
    print("\n========== 2. Load Checkpoint ==========")

    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_DIR,
        local_files_only=True,
        strict=True,
    )

    policy.eval()

    print(f"Checkpoint : {CHECKPOINT_DIR}")
    print(
        f"Device     : "
        f"{next(policy.parameters()).device}"
    )
    print(
        f"Mode       : "
        f"{'train' if policy.training else 'eval'}"
    )
    print(
        f"Chunk size : "
        f"{policy.config.chunk_size}"
    )
    print(
        f"Action dim : "
        f"{policy.config.action_feature.shape[0]}"
    )

    # =========================================================================
    # 3. Processors
    # =========================================================================
    print("\n========== 3. Load Processors ==========")

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

    print("PreProcessor  : PASS")
    print("PostProcessor : PASS")

    # =========================================================================
    # 4. Action Chunk Dataset
    # =========================================================================
    print("\n========== 4. Action Chunk Dataset ==========")

    delta_timestamps = {
        "action": [
            index / metadata.fps
            for index
            in policy.config.action_delta_indices
        ]
    }

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    print(f"Dataset samples : {len(dataset)}")

    # =========================================================================
    # 5. Fixed Evaluation Noise
    # =========================================================================
    print("\n========== 5. Fixed Evaluation Noise ==========")

    torch.manual_seed(1234)

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    device = next(
        policy.parameters()
    ).device

    eval_noise = (
        policy.model.sample_noise(
            noise_shape,
            device,
        )
    )

    print(
        f"Noise shape : "
        f"{tuple(eval_noise.shape)}"
    )

    print(
        f"Noise device: "
        f"{eval_noise.device}"
    )

    # =========================================================================
    # 6. Train Episodes
    # =========================================================================
    print("\n========== 6. Evaluate Train Episodes ==========")

    train_results = []

    for episode_index in TRAIN_EPISODES:

        result = evaluate_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            metadata=metadata,
            episode_index=episode_index,
            eval_noise=eval_noise,
        )

        train_results.append(
            result
        )

        print(
            f"Episode {episode_index} | "
            f"Samples={result['samples']} | "
            f"MAE={result['mae']:.8f}"
        )

    # =========================================================================
    # 7. Validation Episodes
    # =========================================================================
    print("\n========== 7. Evaluate Validation Episodes ==========")

    validation_results = []

    for episode_index in VALIDATION_EPISODES:

        result = evaluate_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            metadata=metadata,
            episode_index=episode_index,
            eval_noise=eval_noise,
        )

        validation_results.append(
            result
        )

        print(
            f"Episode {episode_index} | "
            f"Samples={result['samples']} | "
            f"MAE={result['mae']:.8f}"
        )

    # =========================================================================
    # 8. Overall Train MAE
    # =========================================================================
    print("\n========== 8. Train vs Validation ==========")

    train_error_sum = sum(
        result["abs_error_sum"]
        for result in train_results
    )

    train_valid_values = sum(
        result["valid_values"]
        for result in train_results
    )

    train_mae = (
        train_error_sum
        / train_valid_values
    )

    validation_error_sum = sum(
        result["abs_error_sum"]
        for result in validation_results
    )

    validation_valid_values = sum(
        result["valid_values"]
        for result in validation_results
    )

    validation_mae = (
        validation_error_sum
        / validation_valid_values
    )

    generalization_gap = (
        validation_mae
        - train_mae
    )

    if train_mae > 0:
        validation_ratio = (
            validation_mae
            / train_mae
        )
    else:
        validation_ratio = float("inf")

    print(
        f"Train MAE      : "
        f"{train_mae:.8f}"
    )

    print(
        f"Validation MAE : "
        f"{validation_mae:.8f}"
    )

    print(
        f"Generalization Gap "
        f"(Val - Train) : "
        f"{generalization_gap:+.8f}"
    )

    print(
        f"Val / Train Ratio       : "
        f"{validation_ratio:.4f}"
    )

    # =========================================================================
    # 9. Result
    # =========================================================================
    print("\n========== 9. Result ==========")

    print(
        f"Train Episodes      : "
        f"{TRAIN_EPISODES}"
    )

    print(
        f"Validation Episodes : "
        f"{VALIDATION_EPISODES}"
    )

    print()

    print(
        f"Train MAE      : "
        f"{train_mae:.8f}"
    )

    print(
        f"Validation MAE : "
        f"{validation_mae:.8f}"
    )

    print()

    if validation_mae <= train_mae:
        print(
            "Validation error <= Train error"
        )
    else:
        print(
            "Validation error > Train error"
        )

    # =========================================================================
    # 10. Main Flow
    # =========================================================================
    print("\n========== Main Flow ==========")

    print(
        """
               34_train_ep01
                    │
               Fine-tuned
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    Episode 0 / 1         Episode 2
      Seen Data          Unseen Data
          │                   │
          ▼                   ▼
     Observation          Observation
          │                   │
          ▼                   ▼
      Prediction          Prediction
          │                   │
          ▼                   ▼
    Compare Expert      Compare Expert
          │                   │
          ▼                   ▼
      Train MAE        Validation MAE
          │                   │
          └─────────┬─────────┘
                    ▼
             Generalization Gap
"""
    )

    print("========== DONE ==========")


if __name__ == "__main__":
    main()