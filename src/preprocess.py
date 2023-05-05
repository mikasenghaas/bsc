# preprocess.py
#  by: mika senghaas

import os

from tqdm import tqdm

from config import RAW_DATA_PATH, IMAGE_DATA_PATH, VIDEO_DATA_PATH
from utils import (
    load_preprocess_args,
    load_metadata,
    load_annotations,
    timestamp_to_second,
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
    start_timer = start_task(
        f"Processing raw videos from {args.split} split", get_timer=True
    )

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
        pbar.set_description(f"Extracting {video_id} ({args.split}) : XXX")

        # load metadata
        duration, frame_count = load_metadata(video_path)

        # meta information
        annotations = load_annotations(annotation_path)
        seconds = [timestamp_to_second(x[0]) for x in annotations] + [duration]

        clips_boundaries = [
            (seconds[i], seconds[i + 1]) for i in range(len(seconds) - 1)
        ]

        for clip_id, (start, end) in enumerate(clips_boundaries):
            # extract label depending on clip time
            label = get_label(start, annotations)
            clip_id = str(clip_id+1).zfill(2)

            # create destination directory
            image_destination_path = os.path.join(IMAGE_DATA_PATH, args.split, label)
            video_destination_path = os.path.join(VIDEO_DATA_PATH, args.split, label)
            mkdir(image_destination_path)
            mkdir(video_destination_path)

            # extract images at 1fps for train, 30fps for test
            ffmpeg_extract_images = (
                f"ffmpeg "
                f"-loglevel error "
                f"-ss {start} "
                f"-y "
                f"-i {video_path} "
                f'-vf scale="{args.crop_size}:{args.crop_size}" '
                f"{'-r 5 ' if args.split == 'train' else ''}"
                f"-t {end-start} "
                f"{image_destination_path}/{video_id}_{clip_id}_%03d.jpg"
            )
            pbar.set_description(f"Extracting {video_id} ({args.split}) : IMG ")
            os.system(ffmpeg_extract_images)

            # extract video clips (max. length)
            # sub_seconds = list(range(start, end, args.vid_length)) + [end]
            # sub_bounds = [
            #     (sub_seconds[i], sub_seconds[i + 1])
            #     for i in range(len(sub_seconds) - 1)
            # ]

            # video with 30fps for train and test
            ffmpeg_extract_video = (
                f"ffmpeg "
                f"-loglevel error "
                f"-ss {start} "
                f"-y "
                f"-i {video_path} "
                f'-vf scale="{args.crop_size}:{args.crop_size}, setdar=1:1" '
                f"-t {end-start} "
                f"{video_destination_path}/{video_id}_{clip_id}.mp4"
            )
            pbar.set_description(
                f"Extracting {video_id} ({args.split}) : VID {clip_id}"
            )

            # run command
            os.system(ffmpeg_extract_video)

    end_task("Processing Raw Videos", start_timer)


if __name__ == "__main__":
    main()
