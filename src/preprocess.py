# preprocess.py
#  by: mika senghaas

import os
from tqdm import trange
from pytorchvideo.data.encoded_video import EncodedVideo

from utils import load_video, load_annotations
from config import *

def main():
    # read all directories in raw data folder
    directories = os.listdir(RAW_DATA_PATH)

    # iterate over all videos
    for directory in directories:
        # open video and annotations
        video : EncodedVideo = load_video(directory)
        annotations : Annotation = load_annotations(directory)

        # iterate over video frame by frame
        pbar = trange(10)
        for frame in pbar:
            # get 10x10s clips
            clip = video.get_clip(10*frame, 10*frame+10)

            if clip:
                video_tensor = clip.get('video')
                print(video_tensor.shape)
                # compute label for frame from annotations
                # label = get_label(frame_num, annotations)
                # frame = transform_frame(frame)

                #write_frame(frame, label, frame_num)

        pbar.set_description( f"Extracting {directory}.MOV ({frame_num}/{frame_count})")

if __name__ == "__main__":
    main()
