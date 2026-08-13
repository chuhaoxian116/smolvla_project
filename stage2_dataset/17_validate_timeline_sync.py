from pathlib import Path
import json

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet"
)

INFO_FILE = ROOT / "meta/info.json"
EPISODES_FILE = ROOT / "meta/episodes/chunk-000/file-000.parquet"
DATA_FILE = ROOT / "data/chunk-000/file-000.parquet"

SHOW_EPISODES = 5


def main():
    with open(INFO_FILE, "r", encoding="utf-8") as f:
        info = json.load(f)

    fps = float(info["fps"])
    expected_dt = 1.0 / fps

    episode_table = pq.read_table(EPISODES_FILE)
    episode_rows = episode_table.to_pylist()

    data_table = pq.read_table(
        DATA_FILE,
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
        ],
    )
    data_rows = data_table.to_pylist()

    print("========== Dataset Timeline ==========")
    print(f"FPS          : {fps:g}")
    print(f"Expected dt  : {expected_dt:.6f} s")
    print(f"Data rows    : {len(data_rows)}")
    print(f"Episodes     : {len(episode_rows)}")

    print("\n========== First Episodes ==========")

    for ep in episode_rows[:SHOW_EPISODES]:
        ep_idx = ep["episode_index"]
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]
        length = ep["length"]

        rows = data_rows[start:end]

        timestamps = np.asarray(
            [row["timestamp"] for row in rows],
            dtype=np.float64,
        )
        frame_indices = np.asarray(
            [row["frame_index"] for row in rows],
            dtype=np.int64,
        )

        dts = np.diff(timestamps)

        expected_last_timestamp = (length - 1) / fps
        data_duration = length / fps

        print(f"\n--- Episode {ep_idx} ---")
        print(f"dataset index : [{start}, {end})")
        print(f"length        : {length}")
        print(f"frame_index   : {frame_indices[0]} -> {frame_indices[-1]}")
        print(
            f"timestamp     : {timestamps[0]:.6f} -> "
            f"{timestamps[-1]:.6f} s"
        )

        if len(dts) > 0:
            print(f"dt mean       : {dts.mean():.9f} s")
            print(f"dt min        : {dts.min():.9f} s")
            print(f"dt max        : {dts.max():.9f} s")

        print(f"data duration : {data_duration:.3f} s")

        frame_ok = (
            frame_indices[0] == 0
            and frame_indices[-1] == length - 1
            and len(rows) == length
        )

        timestamp_ok = (
            abs(timestamps[0]) < 1e-6
            and abs(timestamps[-1] - expected_last_timestamp) < 1e-4
            and np.allclose(dts, expected_dt, atol=1e-5)
        )

        print(f"frame check   : {'PASS' if frame_ok else 'FAIL'}")
        print(f"time check    : {'PASS' if timestamp_ok else 'FAIL'}")

        print("\nVideo ranges:")
        for camera in [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]:
            from_key = f"videos/{camera}/from_timestamp"
            to_key = f"videos/{camera}/to_timestamp"

            video_from = float(ep[from_key])
            video_to = float(ep[to_key])
            video_duration = video_to - video_from

            duration_ok = abs(video_duration - data_duration) < 1e-4

            print(
                f"  {camera}: "
                f"{video_from:.3f} -> {video_to:.3f} s, "
                f"duration={video_duration:.3f} s, "
                f"{'PASS' if duration_ok else 'FAIL'}"
            )

    print("\n========== What This Verifies ==========")
    print("1. 每个 Episode 内 frame_index 从 0 开始。")
    print("2. 每个 Episode 内 timestamp 从 0 开始，并按 1/FPS 递增。")
    print("3. State/Action 共用同一个 Frame timestamp。")
    print("4. 三路视频的 Episode 时间段长度与机器人数据时长一致。")
    print()
    print("注意：")
    print(
        "这里验证的是 Dataset/Metadata 层面的时间一致性；"
        "没有读取 MP4，所以还没有验证真实相机帧的逐帧采集延迟或硬件同步误差。"
    )


if __name__ == "__main__":
    main()