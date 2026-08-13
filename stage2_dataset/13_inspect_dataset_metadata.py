from pprint import pprint

from lerobot.datasets import LeRobotDatasetMetadata


REPO_ID = "lerobot/aloha_mobile_cabinet"


def main():
    print("========== Load Dataset Metadata ==========")

    meta = LeRobotDatasetMetadata(REPO_ID)

    print(f"Repo ID        : {REPO_ID}")
    print(f"Total episodes : {meta.total_episodes}")
    print(f"Total frames   : {meta.total_frames}")
    print(f"FPS            : {meta.fps}")
    print(f"Robot type     : {meta.robot_type}")
    print(f"Camera keys    : {meta.camera_keys}")

    print("\n========== Tasks ==========")
    pprint(meta.tasks)

    print("\n========== Features ==========")
    pprint(meta.features)

    print("\n========== Metadata Summary ==========")
    print(meta)


if __name__ == "__main__":
    main()