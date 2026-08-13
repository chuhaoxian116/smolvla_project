import math
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset


FPS = 50
NUM_FRAMES = 20
ACTION_DIM = 6
IMAGE_H = 64
IMAGE_W = 64
EPISODES = 3

TASK = "Move the robot to a target pose."

REPO_ID = "local/mock_multi_episode_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_multi_episode_dataset"
)

RESET_DATASET = True


class MockRobot:
    """模拟机器人：State 逐步跟踪上一周期下发的绝对关节目标。"""

    def __init__(self, initial_offset: float = 0.0):
        self.state = np.full(ACTION_DIM, initial_offset, dtype=np.float32)
        self.target = self.state.copy()

    def get_observation(self, frame_index: int):
        self.state += 0.25 * (self.target - self.state)

        image = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        image[:, :, 0] = frame_index % 256

        return {
            "observation.state": self.state.copy(),
            "observation.images.camera": image,
        }

    def send_action(self, action: np.ndarray):
        self.target = action.astype(np.float32).copy()


class MockExpert:
    """模拟人工专家 / 示教器 / Leader Arm。"""

    def __init__(self, phase: float = 0.0):
        self.phase = phase

    def get_action(self, timestamp: float) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        for i in range(ACTION_DIM):
            action[i] = 0.2 * math.sin(
                timestamp + self.phase + i * 0.2
            )

        return action


def create_dataset():
    if DATASET_ROOT.exists():
        if not RESET_DATASET:
            raise FileExistsError(
                f"Dataset 已存在：{DATASET_ROOT}\n"
                "如需重新创建，请将 RESET_DATASET 改为 True。"
            )

        print(f"Remove old dataset: {DATASET_ROOT}")
        shutil.rmtree(DATASET_ROOT)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
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
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        },
    }

    print("========== Create LeRobotDataset ==========")
    print(f"Repo ID  : {REPO_ID}")
    print(f"Root     : {DATASET_ROOT}")
    print(f"FPS      : {FPS}")
    print(f"Episodes : {EPISODES}")

    return LeRobotDataset.create(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        fps=FPS,
        robot_type="mock_6axis_robot",
        features=features,
        use_videos=False,
    )


def record_episodes(dataset: LeRobotDataset):
    print("\n========== Record Episodes ==========")

    for episode_index in range(EPISODES):
        # 一个 Episode = 一次独立、连续、完整的任务示教。
        # 重新创建 Robot / Expert，模拟：
        # “任务完成 -> 场景复位 -> 下一次示教”。
        robot = MockRobot(initial_offset=0.02 * episode_index)
        expert = MockExpert(phase=0.15 * episode_index)

        print(f"\n--- Episode {episode_index} START ---")

        for frame_index in range(NUM_FRAMES):
            timestamp = frame_index / FPS

            observation = robot.get_observation(frame_index)
            action = expert.get_action(timestamp)

            frame = {
                **observation,
                "action": action,
                "task": TASK,
            }

            dataset.add_frame(frame)
            robot.send_action(action)

            print(
                f"Episode {episode_index} | "
                f"Frame {frame_index:02d} | "
                f"t={timestamp:5.2f}s | "
                f"state[0]={observation['observation.state'][0]: .4f} | "
                f"action[0]={action[0]: .4f}"
            )

        # 当前这一次完整任务示教结束。
        dataset.save_episode()

        print(
            f"--- Episode {episode_index} END: "
            f"save_episode() ---"
        )

    dataset.finalize()

    print("\nAll episodes saved and dataset finalized.")


def inspect_saved_dataset():
    print("\n========== Re-open Dataset ==========")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
    )

    print(dataset)

    print("\n========== Metadata ==========")
    print(f"Episodes : {dataset.meta.total_episodes}")
    print(f"Frames   : {dataset.meta.total_frames}")
    print(f"FPS      : {dataset.meta.fps}")
    print(f"Robot    : {dataset.meta.robot_type}")
    print(f"Cameras  : {dataset.meta.camera_keys}")

    print("\n========== Episode Metadata ==========")

    for episode_index in range(dataset.meta.total_episodes):
        episode = dataset.meta.episodes[episode_index]

        print(
            f"Episode {episode_index}: "
            f"from={episode['dataset_from_index']}, "
            f"to={episode['dataset_to_index']}, "
            f"length={episode['length']}"
        )

    print("\n========== Episode Boundary Frames ==========")

    check_indices = [0, 19, 20, 39, 40, 59]

    for index in check_indices:
        frame = dataset[index]

        print(
            f"global_index={index:02d} | "
            f"episode_index={int(frame['episode_index'])} | "
            f"frame_index={int(frame['frame_index'])} | "
            f"timestamp={float(frame['timestamp']):.2f}s | "
            f"state[0]={float(frame['observation.state'][0]): .4f} | "
            f"action[0]={float(frame['action'][0]): .4f}"
        )

    print("\n========== Expected Relationship ==========")
    print("Global index : 0 ... 59        -> 整个 Dataset 连续")
    print("Episode 0    : frame 0 ... 19  -> 一次完整任务")
    print("Episode 1    : frame 0 ... 19  -> 新的一次完整任务")
    print("Episode 2    : frame 0 ... 19  -> 又一次完整任务")
    print("timestamp    : 每个 Episode 从 0 重新开始")

    print("\n========== Generated Files ==========")

    for path in sorted(DATASET_ROOT.rglob("*")):
        if path.is_file():
            print(path.relative_to(DATASET_ROOT))


def main():
    dataset = create_dataset()
    record_episodes(dataset)
    inspect_saved_dataset()

    print("\n========== Final Concept ==========")
    print("Dataset 全局存储可以连续。")
    print("但每个 Episode 是一条独立、连续、完整的任务轨迹。")
    print()
    print("Episode 内：")
    print("Observation(t) -> Action(t) -> Observation(t+1) 连续")
    print()
    print("Episode 之间：")
    print("通过 save_episode() 建立任务边界，")
    print("不应该被当成同一条连续轨迹。")


if __name__ == "__main__":
    main()