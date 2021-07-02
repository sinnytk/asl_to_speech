
import os
import sys
import cv2
from tqdm import tqdm
from pathlib import Path


CONSTANT = 0
DIRNAME = os.path.dirname(__file__)


def exit_with_error(msg):
    print(msg)
    exit(1)

def split_frames(source_path: Path, dest_path: Path, file):
    video = cv2.VideoCapture(str(source_path)+'/'+file)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # image_folder_name = os.path.splitext(video_path)[0].split('/')[-1]
    # os.mkdir(os.path.join(os.getcwd(), image_folder_name))

    success, image = video.read()
    count = 0
    video_file_name = file.split('.')[0]
    print(f"Writing frames at {dest_path}")
    dest_path = os.path.join(DIRNAME, dest_path)
    pbar = tqdm(total=frames)
    while success:  
        save_file = video_file_name + '_frame_' + str(count) + '.jpg'
        cv2.imwrite(os.path.join(dest_path, save_file), image)
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
    source_path = Path(sys.argv[1])
    # creating default destination path
    dest_path = Path(os.path.dirname(source_path) + '/transformed')

    if (len(sys.argv) == 3):
        dest_path = Path(sys.argv[2])

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if os.path.exists(source_path):
        for file in os.listdir(source_path) :
            if file.endswith(('.mp4','.webm')):  
                split_frames(source_path, dest_path, file)
    else:
        exit_with_error("Video doesn't exist")


if __name__ == "__main__":
    main()
