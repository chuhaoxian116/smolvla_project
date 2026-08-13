from pprint import pprint

from lerobot.datasets import LeRobotDatasetMetadata


REPO_ID = "lerobot/aloha_mobile_cabinet"


def main():
    meta = LeRobotDatasetMetadata(REPO_ID)
    episodes = meta.episodes

    print("========== Dataset ==========")
    print(f"Repo ID  : {REPO_ID}")
    print(f"Episodes : {meta.total_episodes}")
    print(f"Frames   : {meta.total_frames}")
    print(f"FPS      : {meta.fps}")

    print("\n========== Episodes Type ==========")
    print(type(episodes))

    print("\n========== Episodes Columns ==========")
    print(episodes.column_names)

    print("\n========== First 5 Episodes ==========")
    pprint(episodes[:5])

    print("\n========== Episode 0 ==========")
    pprint(episodes[0])


if __name__ == "__main__":
    main()