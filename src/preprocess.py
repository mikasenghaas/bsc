# preprocess.py
#  by: mika senghaas

import os

from tqdm import tqdm

from config import *
from utils import *


def main():
    """
    Main function of the preprocess.py script.

    Preprocesses raw videos in train or test split in the following steps:
    1. Load video into tensor
    2. Load annotations
    3. Extract clips of maximal seconds within the boundaries of labels
    4. Save extracted clips into directory corresponding to label  to disk using ffmpeg
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
    for date in pbar:
        # set paths
        directory = os.path.join(filepath, date)
        video_path = os.path.join(directory, "video.mov")
        annotation_path = os.path.join(directory, "annotations")

        # set progress bar
        pbar.set_description(f"Extracting data/{args.split}/{date}")

        # load metadata
        meta = load_metadata(video_path)
        duration = int(eval(meta["duration"]))

        # meta information
        annotations = load_annotations(annotation_path)
        seconds = [timestamp_to_second(ts) for ts, _ in annotations]

        start_time = 0  # start time of extraction
        clip_num = 0  # number of clips extracted from video
        next_label = 1  # idx of seconds array of next label

        # extract for as long as video is not over
        while start_time < duration:
            # max time allowed (either video duration or max clip length)
            max_time = min(duration, start_time + args.max_length)
            try:
                # get next label time if exists
                next_label_time = seconds[next_label]

                # set end time of extracted clip
                if next_label_time < max_time:
                    end_time = next_label_time
                    next_label += 1
                else:
                    end_time = max_time
            except:
                end_time = max_time

            # extract label depending on clip time
            label = get_label(start_time, annotations)
            destination_dir = os.path.join(
                PROCESSED_DATA_PATH,
                args.split,
                label,
                f"{date}_{str(clip_num).zfill(2)}",
            )
            mkdir(destination_dir)

            # compile ffmpeg commands
            ffmpeg_command = (
                f"ffmpeg "
                f"-loglevel error "
                f"-ss {start_time} "
                f"-y "
                f"-i {video_path} "
                f"-vf scale=224:224 "
                f"-t {end_time-start_time} "
                f"-r {FPS} "
                f"{destination_dir}/%02d.jpg"
            )

            # run command
            os.system(ffmpeg_command)

            # update loop information
            start_time = end_time
            clip_num += 1

    end_task("Processing Raw Videos", start_timer)


if __name__ == "__main__":
    main()
