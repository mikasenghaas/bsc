# preprocess.py
#  by: mika senghaas

import os

from tqdm import tqdm

from config import *
from utils import *

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

    # load args
    args = load_preprocess_args()

    # read all directories in raw data folder
    directories = sorted([d for d in os.listdir(RAW_DATA_PATH) if not d.startswith('.')])

    # iterate over all videos
    pbar = tqdm(directories)
    for date in pbar:
        directory = os.path.join(RAW_DATA_PATH, date)
        video_path = os.path.join(directory, "video.mov")
        annotation_path = os.path.join(directory, "annotations")

        # set progress bar
        pbar.set_description(f"Extracting {video_path}")

        # load metadata
        meta : dict = load_metadata(video_path)
        duration : int = int(eval(meta['duration']))

        # meta information
        annotations = load_annotations(annotation_path)
        seconds = [timestamp_to_second(ts) for ts, _ in annotations]

        # iterate over clips
        # pbar = trange(0, int(duration)-CLIP_LENGTH, CLIP_LENGTH)
        # total_clips = (int(duration)-CLIP_LENGTH) // CLIP_LENGTH
        start_time = 0
        clip_num = 0
        next_label = 1 # idx of seconds array of next label
        while start_time < duration:
            # pbar.set_description(f"Extracting {directory}/video.mov ({clip_num+1}/{total_clips})")
            end_time = min(duration, start_time + MAX_LENGTH)
            if next_label < len(seconds):
                end_time = min(end_time, seconds[next_label])
                if end_time == seconds[next_label]:
                    next_label += 1

            # extract label depending on clip time
            label = get_label(start_time, annotations)
            destination_dir = os.path.join(PROCESSED_DATA_PATH, label, f"{date}_{str(clip_num).zfill(2)}")
            mkdir(destination_dir)

            # run ffmpeg
            ffmpeg_command = f"ffmpeg -loglevel error -ss {start_time} -y -i {video_path} -vf scale=224:224 -t {end_time-start_time} -r {FPS} {destination_dir}/%02d.jpg"
            os.system(ffmpeg_command)

            # update loop information
            start_time = end_time
            clip_num += 1

    end_task("Processing Raw Videos", start_timer)

if __name__ == "__main__":
    main()
