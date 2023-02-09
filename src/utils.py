# utils.py
#  by: mika senghaas

# import torchvision
from matplotlib import pyplot as plt
from pytorchvideo.data.encoded_video import EncodedVideo

from config import *

def load_video(date : str) -> EncodedVideo:
    filepath = f"{RAW_DATA_PATH}/{date}/{date}.MOV"
    return EncodedVideo.from_path(filepath, decode_audio=False)

def load_annotations(date : str) -> Annotation:
    targets  = []
    filepath = f"{RAW_DATA_PATH}/{date}/annotations"
    with open(filepath, "r") as file:
        for line in file:
            timestamp, label = line.strip().split(',')
            targets.append((timestamp, label))

    return targets 

def timestamp_to_frame(timestamp : str) -> int:
  mm, ss = map(int, timestamp.split(':'))
  return 30 * (mm * 60 + ss)

def get_label(frame_num: int, annotations: list[tuple[str, str]]) -> str:
  # assumes targets is sorted by timestamp
  prev_label = annotations[0][1]
  for i in range(1, len(annotations)):
    timestamp, label = annotations[i]
    framecount = timestamp_to_frame(timestamp)
    if frame_num < framecount:
      return prev_label
    prev_label = label

  return annotations[-1][1]

def visualize_batch(batch, batch_size, classes, dataset_type):
  # initialize a figure
  fig = plt.figure("{} batch".format(dataset_type), figsize=(batch_size, batch_size))
  # loop over the batch size
  for i in range(0, batch_size):
    # create a subplot
    ax = plt.subplot(2, 4, i + 1)
    # grab the image, convert it from channels first ordering to
    # channels last ordering, and scale the raw pixel intensities
    # to the range [0, 255]
    image = batch[0][i].cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = (image * 255.0).astype("uint8")
    # grab the label id and get the label from the classes list
    idx = batch[1][i]
    label = classes[idx]
    # show the image along with the label
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
  # show the plot
  plt.tight_layout()
  plt.show()
