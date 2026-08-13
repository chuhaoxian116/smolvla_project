from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset


REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

CHUNK_SIZE = 10
BATCH_SIZE = 4


def print_sample(dataset: LeRobotDataset, index: int):
    sample = dataset[index]

    print(f"\n========== Sample global_index={index} ==========")
    print(f"episode_index : {int(sample['episode_index'])}")
    print(f"frame_index   : {int(sample['frame_index'])}")
    print(f"timestamp     : {float(sample['timestamp']):.2f}s")

    print(f"state shape   : {tuple(sample['observation.state'].shape)}")
    print(f"image shape   : {tuple(sample['observation.images.camera'].shape)}")
    print(f"action shape  : {tuple(sample['action'].shape)}")

    action_is_pad = sample.get("action_is_pad")

    if action_is_pad is not None:
        print(f"action_is_pad : {action_is_pad.tolist()}")

    print("\nAction chunk, joint_1:")
    for t in range(sample["action"].shape[0]):
        pad_text = ""

        if action_is_pad is not None:
            pad_text = " PAD" if bool(action_is_pad[t]) else " VALID"

        print(
            f"  chunk[{t:02d}] "
            f"action[0]={float(sample['action'][t, 0]): .6f}"
            f"{pad_text}"
        )


def main():
    print("========== Build Action Chunk Dataset ==========")

    delta_timestamps = {
        "action": [
            t / 50
            for t in range(CHUNK_SIZE)
        ]
    }

    print(f"Chunk size       : {CHUNK_SIZE}")
    print(f"delta_timestamps : {delta_timestamps['action']}")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    print(f"Dataset FPS      : {dataset.fps}")
    print(f"Dataset frames   : {len(dataset)}")

    # Episode 0 = global index [0, 20)
    # index 5 -> Action 5 ... 14，全部位于 Episode 0。
    # index 15 -> 请求 Action 15 ... 24，但 Episode 0 在 19 结束。
    # Episode 外的部分应该被标记为 padding，不能跨 Episode。
    print_sample(dataset, 5)
    print_sample(dataset, 15)

    print("\n========== DataLoader ==========")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    print(f"batch state shape         : {tuple(batch['observation.state'].shape)}")
    print(f"batch image shape         : {tuple(batch['observation.images.camera'].shape)}")
    print(f"batch action shape        : {tuple(batch['action'].shape)}")

    if "action_is_pad" in batch:
        print(f"batch action_is_pad shape : {tuple(batch['action_is_pad'].shape)}")

    print("\n========== Shape Meaning ==========")
    print("单个 Frame 原始 Action       : [6]")
    print(f"delta_timestamps 后 Action   : [{CHUNK_SIZE}, 6]")
    print(
        f"DataLoader 后 Action         : "
        f"[{BATCH_SIZE}, {CHUNK_SIZE}, 6]"
    )

    print("\n========== Model Handoff ==========")
    print("LeRobotDataset")
    print("      ↓ delta_timestamps")
    print("Training Sample")
    print("  observation.state       [6]")
    print("  observation.images      [3,64,64]")
    print(f"  expert action chunk     [{CHUNK_SIZE},6]")
    print(f"  action_is_pad           [{CHUNK_SIZE}]")
    print("      ↓ DataLoader")
    print(f"Batch action              [{BATCH_SIZE},{CHUNK_SIZE},6]")
    print("      ↓ PreProcessor")
    print("      ↓ SmolVLA.forward(batch)")
    print("      ↓ Loss / Backpropagation")


if __name__ == "__main__":
    main()