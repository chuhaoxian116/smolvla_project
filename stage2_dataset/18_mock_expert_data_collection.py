import math
from dataclasses import dataclass

import numpy as np


FPS = 50
DT = 1.0 / FPS
NUM_FRAMES = 20
ACTION_DIM = 6
TASK = "Move the robot to a target pose."


@dataclass
class MockObservation:
    state: np.ndarray
    image: np.ndarray


class MockRobot:
    """模拟机器人：State 会逐步跟踪上一周期下发的绝对 Action。"""

    def __init__(self):
        self.state = np.zeros(ACTION_DIM, dtype=np.float32)
        self.target = self.state.copy()

    def get_observation(self) -> MockObservation:
        # 模拟机器人实际状态逐步靠近目标，而不是瞬间等于 Action。
        self.state += 0.25 * (self.target - self.state)

        # 模拟一帧 RGB 图像，只关心 Shape，不关心真实内容。
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        return MockObservation(
            state=self.state.copy(),
            image=image,
        )

    def send_action(self, action: np.ndarray):
        # 当前 Demo 中 action 定义为“绝对关节目标”。
        self.target = action.astype(np.float32).copy()


class MockExpert:
    """模拟专家：根据时间产生连续的绝对关节目标。"""

    def get_action(self, timestamp: float) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        for i in range(ACTION_DIM):
            action[i] = 0.2 * math.sin(timestamp + i * 0.2)

        return action


def collect_episode():
    robot = MockRobot()
    expert = MockExpert()

    frames = []

    print("========== Mock Data Collection ==========")
    print(f"FPS        : {FPS}")
    print(f"DT         : {DT:.3f} s")
    print(f"Num frames : {NUM_FRAMES}")
    print(f"Action dim : {ACTION_DIM}")
    print(f"Task       : {TASK}")

    for frame_index in range(NUM_FRAMES):
        timestamp = frame_index / FPS

        # 1. 读取当前 Observation。
        observation = robot.get_observation()

        # 2. Expert 根据当前任务/时间产生绝对 Action。
        expert_action = expert.get_action(timestamp)

        # 3. 把“当前 Observation + 本周期 Expert Action”组成一个训练 Frame。
        frame = {
            "observation.state": observation.state.copy(),
            "observation.images.camera": observation.image.copy(),
            "action": expert_action.copy(),
            "timestamp": timestamp,
            "episode_index": 0,
            "frame_index": frame_index,
            "task": TASK,
            "next.done": frame_index == NUM_FRAMES - 1,
        }

        frames.append(frame)

        # 4. 将与 Dataset 中记录完全相同的 Action 发给机器人。
        robot.send_action(expert_action)

        print(
            f"Frame {frame_index:02d} | "
            f"t={timestamp:5.2f}s | "
            f"state[0]={observation.state[0]: .4f} | "
            f"action[0]={expert_action[0]: .4f} | "
            f"done={frame['next.done']}"
        )

    return frames


def inspect_frames(frames):
    print("\n========== Inspect Collected Frames ==========")

    for i in [0, 1, len(frames) - 1]:
        frame = frames[i]

        print(f"\n--- Frame {i} ---")
        print(f"timestamp     : {frame['timestamp']:.3f}")
        print(f"episode_index : {frame['episode_index']}")
        print(f"frame_index   : {frame['frame_index']}")
        print(f"next.done     : {frame['next.done']}")
        print(f"state shape   : {frame['observation.state'].shape}")
        print(f"image shape   : {frame['observation.images.camera'].shape}")
        print(f"action shape  : {frame['action'].shape}")
        print(f"state         : {np.round(frame['observation.state'], 4)}")
        print(f"action        : {np.round(frame['action'], 4)}")


def main():
    frames = collect_episode()
    inspect_frames(frames)

    print("\n========== Data Flow ==========")
    print("Robot.get_observation()")
    print("        ↓")
    print("Observation(State + Camera)")
    print("        ↓")
    print("Expert.get_action()")
    print("        ↓")
    print("Absolute Expert Action")
    print("        ├──> Dataset Frame")
    print("        └──> Robot.send_action()")

    print("\nConclusion:")
    print("一个训练 Frame 记录的是：")
    print("当前 Observation(t) + 本周期真正执行的 Expert Action(t)。")


if __name__ == "__main__":
    main()