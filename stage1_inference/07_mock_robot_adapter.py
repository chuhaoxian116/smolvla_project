import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"
ACTION_DIM = 6


class MockRobotAdapter:
    """
    假机器人适配器。

    当前不连接真实机器人，只模拟：
    SmolVLA Action -> Robot Adapter -> Robot
    """

    def __init__(self, action_dim: int):
        # 期望每个 Action 的维度，例如 6 维
        self.action_dim = action_dim

        # 保存最后一次收到的 Action，方便测试
        self.last_action = None

    def send_action(self, action: torch.Tensor):
        """
        模拟向机器人发送一个 Action。

        输入 action:
            shape = [6]
        """

        # 检查 Action 是否是一维
        if action.ndim != 1:
            raise ValueError(
                f"Action shape error: expected 1-D, got {tuple(action.shape)}"
            )

        # 检查 Action 维度是否符合预期
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Action dim error: expected {self.action_dim}, "
                f"got {action.shape[0]}"
            )

        # 模拟机器人已经收到这个 Action
        self.last_action = action.clone()

        print(f"MockRobot <- {action}")

    def execute_chunk(self, action_chunk: torch.Tensor):
        """
        依次执行整个 Action Chunk。

        输入：
            action_chunk shape = [50, 6]
        """

        print(
            f"\nExecute Action Chunk: "
            f"{action_chunk.shape[0]} actions"
        )

        # 每次从 Chunk 中取出一个 [6] Action
        for i in range(action_chunk.shape[0]):
            action = action_chunk[i]

            print(f"Step[{i:02d}] ", end="")
            self.send_action(action)


def build_fake_observation():
    """
    构造假的 Observation：
    6维 State + 三张假图片 + Task
    """

    return {
        "observation.state": torch.tensor(
            [0.10, -0.20, 0.30, -0.40, 0.50, 0.00],
            dtype=torch.float32,
        ),

        # 三张全黑 RGB 图片
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
    # 固定随机种子，方便重复测试时输出更稳定
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("===== 1. Load Model =====")

    # 加载 SmolVLA
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.eval()

    # 创建输入前处理器和输出后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        MODEL_ID,
    )

    print("\n===== 2. Build Observation =====")

    # 构造假的机器人 Observation
    observation = build_fake_observation()

    print("State :", observation["observation.state"])
    print("Task  :", observation["task"])

    print("\n===== 3. PreProcess =====")

    # 原始 Observation
    # -> Batch / Tokenizer / CUDA / Normalization
    processed = preprocessor(observation)

    print("State :", processed["observation.state"].shape)
    print("Camera:", processed["observation.images.camera1"].shape)
    print("Tokens:", processed["observation.language.tokens"].shape)

    print("\n===== 4. SmolVLA Predict =====")

    # SmolVLA 一次输出完整 Action Chunk
    with torch.inference_mode():
        model_action_chunk = policy.predict_action_chunk(
            processed
        )

    print(
        "Model Action Chunk:",
        tuple(model_action_chunk.shape),
        model_action_chunk.device,
    )

    print("\n===== 5. PostProcess =====")

    # 模型输出 -> PostProcessor
    # 当前主要看到 CUDA -> CPU
    postprocessed_action_chunk = postprocessor(
        model_action_chunk
    )

    print(
        "Postprocessed Action Chunk:",
        tuple(postprocessed_action_chunk.shape),
        postprocessed_action_chunk.device,
    )

    print("\n===== 6. Robot Adapter =====")

    # 创建假的机器人执行层
    robot = MockRobotAdapter(
        action_dim=ACTION_DIM
    )

    # 原始 shape:
    # [1, 50, 6]
    #
    # 去掉 Batch 后：
    # [50, 6]
    action_chunk = postprocessed_action_chunk[0]

    # 模拟把 50 个 Action 按顺序交给机器人
    robot.execute_chunk(action_chunk)

    print("\n===== 7. Result =====")

    print("Last Action:")
    print(robot.last_action)

    print("\nMock Robot Adapter Test: PASS")


if __name__ == "__main__":
    main()