# preprocess.py
#  by: mika senghaas

import os

from tqdm import tqdm

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)
from utils import (
    load_preprocess_args,
    load_metadata,
    load_annotations,
    get_label,
    start_task,
    end_task,
    mkdir,
    ls,
)


def main():
    """
    Main function of the preprocess.py script.

    Preprocesses raw videos in the filepath `src/data/raw/train` or `src/data/raw/test`
    and saves the extracted clips into the filepath `src/data/processed/train` or
    `src/data/processed/test`. The clips are saved into a directory
    corresponding with the directory name and the labels are saved into a file
    """
    # load args
    args = load_preprocess_args()

    # start task
    start_timer = start_task("Processing Raw Videos", get_timer=True)

    # read all directories in raw data folder
    filepath = os.path.join(RAW_DATA_PATH, args.split)
    directories = ls(filepath)

    # iterate over all videos
    pbar = tqdm(directories)
    for video_id in pbar:
        # set paths
        directory = os.path.join(filepath, video_id)
        video_path = os.path.join(directory, "video.mov")
        annotation_path = os.path.join(directory, "annotations")

        # set progress bar
        pbar.set_description(f"Extracting data/{args.split}/{video_id}")

        # load metadata
        duration, frame_count = load_metadata(video_path)

        # meta information
        annotations = load_annotations(annotation_path)

        # make processed filepath
        savepath = os.path.join(PROCESSED_DATA_PATH, args.split, video_id)
        mkdir(savepath)

        # write labels of extracted frames to file
        labels_path = os.path.join(savepath, "labels.txt")
        with open(labels_path, "w") as f:
            for second in range(duration):
                for _ in range(args.fps):
                    label = get_label(second, annotations)
                    f.write(f"{label}\n")

        # compile ffmpeg command to extract frames
        ffmpeg_command = (
            f"ffmpeg "
            f"-loglevel error "
            f"-y "
            f"-i {video_path} "
            f"-vf scale=224:224 "
            f"-t {duration} "
            f"-r {args.fps} "
            f"{savepath}/%03d.jpg"
        )

        # run command
        os.system(ffmpeg_command)

    end_task("Processing Raw Videos", start_timer)


if __name__ == "__main__":
    main()
