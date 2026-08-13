from pathlib import Path
from pprint import pprint

from lerobot.datasets import LeRobotDataset


REPO_ID = "local/mock_robot_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_robot_dataset"
)


def print_frame(dataset: LeRobotDataset, index: int):
    frame = dataset[index]

    print(f"\n========== Global Frame {index} ==========")
    print(f"episode_index : {frame['episode_index']}")
    print(f"frame_index   : {frame['frame_index']}")
    print(f"timestamp     : {frame['timestamp']}")
    print(f"task_index    : {frame['task_index']}")

    print(f"state shape   : {tuple(frame['observation.state'].shape)}")
    print(f"image shape   : {tuple(frame['observation.images.camera'].shape)}")
    print(f"action shape  : {tuple(frame['action'].shape)}")

    print("\nstate:")
    print(frame["observation.state"])

    print("\naction:")
    print(frame["action"])


def main():
    print("========== Open Own Dataset ==========")
    print(f"Repo ID : {REPO_ID}")
    print(f"Root    : {DATASET_ROOT}")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    meta = dataset.meta

    print("\n========== Metadata ==========")
    print(f"Total episodes : {meta.total_episodes}")
    print(f"Total frames   : {meta.total_frames}")
    print(f"FPS            : {meta.fps}")
    print(f"Robot type     : {meta.robot_type}")
    print(f"Camera keys    : {meta.camera_keys}")

    print("\n========== Tasks ==========")
    pprint(meta.tasks)

    print("\n========== Features ==========")
    pprint(meta.features)

    print("\n========== Episode 0 Metadata ==========")
    episode0 = meta.episodes[0]
    pprint(episode0)

    print("\n========== Episode 0 Summary ==========")
    print(f"episode_index      : {episode0['episode_index']}")
    print(f"dataset_from_index : {episode0['dataset_from_index']}")
    print(f"dataset_to_index   : {episode0['dataset_to_index']}")
    print(f"length             : {episode0['length']}")
    print(f"tasks              : {episode0['tasks']}")

    # 我们自己创建的数据集只有 20 帧。
    print_frame(dataset, 0)
    print_frame(dataset, 1)
    print_frame(dataset, len(dataset) - 1)

    print("\n========== Timeline Check ==========")
    frame0 = dataset[0]
    frame1 = dataset[1]
    last = dataset[len(dataset) - 1]

    dt = float(frame1["timestamp"] - frame0["timestamp"])

    print(f"Frame 0 timestamp : {frame0['timestamp']}")
    print(f"Frame 1 timestamp : {frame1['timestamp']}")
    print(f"Measured dt       : {dt:.6f} s")
    print(f"Expected dt       : {1.0 / meta.fps:.6f} s")
    print(f"Last timestamp    : {last['timestamp']}")

    print("\n========== Conclusion ==========")
    print("我们已经完成：")
    print("Create -> add_frame -> save_episode -> finalize -> reopen -> inspect")
    print()
    print("自己的 LeRobot Dataset 读写闭环已经建立。")


if __name__ == "__main__":
    main()