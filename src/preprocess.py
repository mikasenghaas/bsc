# preprocess.py
#  by: mika senghaas

import os

from tqdm import trange

from config import *
from utils import (
    clip_video,
    end_task,
    get_label,
    load_annotations,
    load_metadata,
    load_video,
    start_task,
)

def main():
    """
    Load annotations and video from RAW_DATA_PATH directory where each video-annotation pairing
    is saved in a directory with the date of recording called `DDMMYY`. The video is called 
    `video.mov` and the annotations is a text file called `annotations`.

    The video and annotations are loaded and depending split into clips of length (s)
    CLIP_LENGTH. For each clip, the label (location denoting floor and corner at ITU) is extracted,
    so that the clip can be saved into PROCESSED_DATA_PATH into a directory with the name of the label
    and the clip called `DDMMYY_XY.mov` where `XY` is the clip number.

    Efficiency: ~7s for extrating 1 labelled 10s clip
    """
    start_timer = start_task("Processing Raw Videos", get_timer=True)

    # read all directories in raw data folder
    directories = os.listdir(RAW_DATA_PATH)

    # iterate over all videos
    for date in directories:
        directory = f"{RAW_DATA_PATH}/{date}"
        video_path = f"{directory}/video.mov"
        annotation_path = f"{directory}/annotations"

        # load metadata
        meta : dict = load_metadata(video_path)
        duration : int = eval(meta['duration'])

        # load video and meta information
        video = load_video(video_path)
        annotations : Annotation = load_annotations(annotation_path)

        # iterate over clips
        pbar = trange(0, int(duration)-CLIP_LENGTH, CLIP_LENGTH)
        total_clips = (int(duration)-CLIP_LENGTH) // CLIP_LENGTH
        for clip_num, start_time in enumerate(pbar):
            pbar.set_description(f"Extracting {directory}.MOV ({clip_num+1}/{total_clips})")
            end_time = start_time+CLIP_LENGTH

            # extract label depending on clip time
            label = get_label(start_time + CLIP_LENGTH//2, annotations)
            destination_path = f"{PROCESSED_DATA_PATH}/{label}/{date}_{str(clip_num).zfill(2)}.mov"

            # save clip from video according to label and start/ end time
            clip_video(video, dst=destination_path, start_time=start_time, end_time=end_time)

        video.close()
        end_task("Processing Raw Videos", start_timer)

if __name__ == "__main__":
    main()
