from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_FILE = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "cache/huggingface/lerobot/lerobot/"
    "aloha_mobile_cabinet/data/chunk-000/file-000.parquet"
)

# 只分析 Episode 0 的前 N 帧，避免跨 Episode 比较。
NUM_FRAMES = 200


def main():
    columns = [
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
    ]

    table = pq.read_table(DATA_FILE, columns=columns)
    rows = table.to_pylist()

    episode0 = [row for row in rows if row["episode_index"] == 0]
    episode0 = episode0[:NUM_FRAMES]

    if len(episode0) < 2:
        raise RuntimeError("Episode 0 数据不足，至少需要 2 帧。")

    states = np.asarray(
        [row["observation.state"] for row in episode0],
        dtype=np.float32,
    )
    actions = np.asarray(
        [row["action"] for row in episode0],
        dtype=np.float32,
    )
    timestamps = np.asarray(
        [row["timestamp"] for row in episode0],
        dtype=np.float32,
    )

    # t = 0 ... N-2
    state_t = states[:-1]
    state_t1 = states[1:]
    action_t = actions[:-1]
    action_t1 = actions[1:]

    # 1. Action(t) 与当前 State(t) 的差异
    err_action_state_t = np.abs(action_t - state_t)

    # 2. Action(t) 与下一帧 State(t+1) 的差异
    err_action_state_t1 = np.abs(action_t - state_t1)

    # 3. 机器人状态每 20 ms 实际变化量
    state_step = np.abs(state_t1 - state_t)

    # 4. 专家 Action 每 20 ms 的变化量
    action_step = np.abs(action_t1 - action_t)

    print("========== Dataset ==========")
    print(f"File          : {DATA_FILE}")
    print(f"Episode       : 0")
    print(f"Frames used   : {len(episode0)}")
    print(f"Action dim    : {actions.shape[1]}")
    print(f"Time range    : {timestamps[0]:.3f}s -> {timestamps[-1]:.3f}s")

    print("\n========== Overall Comparison ==========")
    print(
        "mean |Action(t) - State(t)|     : "
        f"{err_action_state_t.mean():.6f}"
    )
    print(
        "mean |Action(t) - State(t+1)|   : "
        f"{err_action_state_t1.mean():.6f}"
    )
    print(
        "mean |State(t+1) - State(t)|    : "
        f"{state_step.mean():.6f}"
    )
    print(
        "mean |Action(t+1) - Action(t)|  : "
        f"{action_step.mean():.6f}"
    )

    print("\n========== Max Difference ==========")
    print(
        "max  |Action(t) - State(t)|     : "
        f"{err_action_state_t.max():.6f}"
    )
    print(
        "max  |Action(t) - State(t+1)|   : "
        f"{err_action_state_t1.max():.6f}"
    )
    print(
        "max  |State(t+1) - State(t)|    : "
        f"{state_step.max():.6f}"
    )
    print(
        "max  |Action(t+1) - Action(t)|  : "
        f"{action_step.max():.6f}"
    )

    print("\n========== First 5 Frame Pairs ==========")
    for i in range(min(5, len(episode0) - 1)):
        print(
            f"\nFrame {i} -> {i + 1} "
            f"({timestamps[i]:.3f}s -> {timestamps[i + 1]:.3f}s)"
        )

        print(
            "mean |Action(t) - State(t)|   = "
            f"{err_action_state_t[i].mean():.6f}"
        )
        print(
            "mean |Action(t) - State(t+1)| = "
            f"{err_action_state_t1[i].mean():.6f}"
        )
        print(
            "mean |State(t+1) - State(t)|  = "
            f"{state_step[i].mean():.6f}"
        )

    print("\n========== Per-Dimension Mean ==========")
    for dim in range(actions.shape[1]):
        print(
            f"dim {dim:02d}: "
            f"|A-S(t)|={err_action_state_t[:, dim].mean():.6f}, "
            f"|A-S(t+1)|={err_action_state_t1[:, dim].mean():.6f}, "
            f"|dState|={state_step[:, dim].mean():.6f}, "
            f"|dAction|={action_step[:, dim].mean():.6f}"
        )

    print("\n========== Reading Guide ==========")
    print(
        "1) 如果 Action(t) 和 State(t) / State(t+1) 数值处于相同量级，"
        "说明它更像同一物理空间中的目标量。"
    )
    print(
        "2) 如果 Action 是 Delta，通常应重点观察它是否更像“变化量”；"
        "本脚本只做数值比较，不直接判定语义。"
    )
    print(
        "3) 是否为绝对关节目标、单位是什么，仍需结合 Dataset 定义确认。"
    )


if __name__ == "__main__":
    main()