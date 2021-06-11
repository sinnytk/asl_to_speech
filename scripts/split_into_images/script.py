
import os
import sys
import cv2
from tqdm import tqdm
from pathlib import Path



def exit_with_error(msg):
    print(msg)
    exit(1)


def split_frames(video_path, dest_path):
    video = cv2.VideoCapture(str(video_path))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # image_folder_name = os.path.splitext(video_path)[0].split('/')[-1]
    # os.mkdir(os.path.join(os.getcwd(), image_folder_name))

    success, image = video.read()
    count = 0
    print(f"Writing frames at {dest_path}")

    pbar = tqdm(total=frames)
    while success:
        cv2.imwrite(f"{dest_path}/frame-{count}.jpg", image)
        success, image = video.read()
        count += 1
        pbar.update(1)
    pbar.close()


def main():
    try:
        assert len(sys.argv) >= 2
    except AssertionError:
        exit_with_error(
            "Script expected ONLY two command line arguments" +
            "\nMake sure you inserted video path")
    # creating video path
    video_path = Path(sys.argv[1])
    # creating default destination path
    dest_path = Path(os.path.dirname(video_path) + '/transformed')

    if (len(sys.argv) == 3):
        dest_path = Path(sys.argv[2])

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    if os.path.exists(video_path):
        split_frames(video_path, dest_path)
    else:
        exit_with_error("Video doesn't exist")


if __name__ == "__main__":
    main()
