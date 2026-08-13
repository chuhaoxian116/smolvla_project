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

TASK = "Move the robot to a target pose."

REPO_ID = "local/mock_robot_dataset"

DATASET_ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "datasets/mock_robot_dataset"
)

# 为了方便重复学习运行：
# True  = 如果目录已存在，删除后重新创建
# False = 已存在时直接报错，避免覆盖
RESET_DATASET = True


class MockRobot:
    """模拟机器人：当前 State 逐步跟踪上一周期下发的绝对关节目标。"""

    def __init__(self):
        self.state = np.zeros(ACTION_DIM, dtype=np.float32)
        self.target = self.state.copy()

    def get_observation(self, frame_index: int):
        # 模拟实际机器人不会瞬间到达 Action，而是逐步跟踪。
        self.state += 0.25 * (self.target - self.state)

        # 模拟 RGB Camera。
        # 使用 uint8 HWC: [H, W, C]
        image = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

        # 让图像随时间变化，方便后续确认每一帧不是完全一样。
        value = frame_index % 256
        image[:, :, 0] = value

        return {
            "observation.state": self.state.copy(),
            "observation.images.camera": image,
        }

    def send_action(self, action: np.ndarray):
        # 本 Demo 的 Action 语义：
        # 绝对 Joint Target。
        self.target = action.astype(np.float32).copy()


class MockExpert:
    """模拟人工专家 / 示教器 / Leader Arm，产生绝对关节目标。"""

    def get_action(self, timestamp: float) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        for i in range(ACTION_DIM):
            action[i] = 0.2 * math.sin(timestamp + i * 0.2)

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
    print(f"Repo ID : {REPO_ID}")
    print(f"Root    : {DATASET_ROOT}")
    print(f"FPS     : {FPS}")

    # use_videos=False:
    # 第一版先使用 image-backed Dataset，
    # 避免一开始引入 MP4 编码流程。
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=DATASET_ROOT,
        fps=FPS,
        robot_type="mock_6axis_robot",
        features=features,
        use_videos=False,
    )

    return dataset


def record_one_episode(dataset: LeRobotDataset):
    robot = MockRobot()
    expert = MockExpert()

    print("\n========== Record Episode 0 ==========")

    for frame_index in range(NUM_FRAMES):
        timestamp = frame_index / FPS

        # 1. 读取当前机器人 Observation。
        observation = robot.get_observation(frame_index)

        # 2. Expert 产生当前周期的绝对 Action。
        action = expert.get_action(timestamp)

        # 3. 组成 LeRobot Frame。
        #
        # 注意：
        # episode_index / frame_index / timestamp / next.done
        # 不需要我们自己塞进去，LeRobotDataset 会根据采集过程生成。
        frame = {
            **observation,
            "action": action,
            "task": TASK,
        }

        # 4. 真正写入 LeRobot 当前 Episode Buffer。
        dataset.add_frame(frame)

        # 5. 同一个 Action 下发给 MockRobot。
        robot.send_action(action)

        print(
            f"Frame {frame_index:02d} | "
            f"t={timestamp:5.2f}s | "
            f"state[0]={observation['observation.state'][0]: .4f} | "
            f"action[0]={action[0]: .4f}"
        )

    # 把当前 Episode Buffer 真正保存到磁盘。
    dataset.save_episode()

    # Dataset v3 写完后必须 finalize，
    # 用来刷新 metadata、关闭 parquet writer。
    dataset.finalize()

    print("\nEpisode saved and dataset finalized.")


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

    print("\n========== First Frame ==========")
    frame0 = dataset[0]

    print(f"episode_index : {frame0['episode_index']}")
    print(f"frame_index   : {frame0['frame_index']}")
    print(f"timestamp     : {frame0['timestamp']}")
    print(f"state shape   : {tuple(frame0['observation.state'].shape)}")
    print(f"image shape   : {tuple(frame0['observation.images.camera'].shape)}")
    print(f"action shape  : {tuple(frame0['action'].shape)}")
    print(f"task_index    : {frame0['task_index']}")

    print("\nstate:")
    print(frame0["observation.state"])

    print("\naction:")
    print(frame0["action"])

    print("\n========== Generated Files ==========")

    for path in sorted(DATASET_ROOT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(DATASET_ROOT)
            print(rel)


def main():
    dataset = create_dataset()
    record_one_episode(dataset)
    inspect_saved_dataset()

    print("\n========== Final Data Flow ==========")
    print("MockRobot.get_observation()")
    print("        +")
    print("MockExpert.get_action()")
    print("        ↓")
    print("LeRobotDataset.add_frame()")
    print("        ↓")
    print("LeRobotDataset.save_episode()")
    print("        ↓")
    print("LeRobotDataset.finalize()")
    print("        ↓")
    print(DATASET_ROOT)


if __name__ == "__main__":
    main()