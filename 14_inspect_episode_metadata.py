from lerobot.datasets import LeRobotDatasetMetadata


REPO_ID = "lerobot/aloha_mobile_cabinet"


def main():
    meta = LeRobotDatasetMetadata(REPO_ID)

    print("========== Dataset ==========")
    print(f"Repo ID  : {REPO_ID}")
    print(f"Episodes : {meta.total_episodes}")
    print(f"Frames   : {meta.total_frames}")
    print(f"FPS      : {meta.fps}")

    print("\n========== Episodes Type ==========")
    print(type(meta.episodes))

    print("\n========== Episodes Columns ==========")
    print(meta.episodes.columns.tolist())

    print("\n========== First 5 Episodes ==========")
    print(meta.episodes.head())

    print("\n========== Episode 0 ==========")
    print(meta.episodes.iloc[0])


if __name__ == "__main__":
    main()