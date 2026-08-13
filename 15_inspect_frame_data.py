from pathlib import Path
from pprint import pprint

import pyarrow.parquet as pq


DATA_FILE = Path(
    "/home/smartrobot/Documents/smolvla_workspace/"
    "cache/huggingface/lerobot/lerobot/"
    "aloha_mobile_cabinet/data/chunk-000/file-000.parquet"
)


def print_frame(table, index):
    """打印指定全局 Frame 的核心数据。"""
    row = table.slice(index, 1).to_pylist()[0]

    print(f"\n========== Global Frame {index} ==========")
    print(f"episode_index : {row['episode_index']}")
    print(f"frame_index   : {row['frame_index']}")
    print(f"timestamp     : {row['timestamp']}")
    print(f"task_index    : {row['task_index']}")
    print(f"next.done     : {row['next.done']}")

    print("\nobservation.state:")
    pprint(row["observation.state"])

    print("\naction:")
    pprint(row["action"])


def main():
    print("========== Parquet File ==========")
    print(DATA_FILE)

    parquet = pq.ParquetFile(DATA_FILE)

    print("\n========== Schema ==========")
    print(parquet.schema)

    # 这里只读取机器人数据，不读取视频。
    columns = [
        "observation.state",
        "observation.effort",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.done",
        "index",
        "task_index",
    ]

    table = pq.read_table(DATA_FILE, columns=columns)

    print("\n========== Table ==========")
    print(f"Rows    : {table.num_rows}")
    print(f"Columns : {table.num_columns}")

    # Episode 0 开头
    print_frame(table, 0)
    print_frame(table, 1)

    # Episode 0 → Episode 1 边界
    print_frame(table, 1499)
    print_frame(table, 1500)


if __name__ == "__main__":
    main()