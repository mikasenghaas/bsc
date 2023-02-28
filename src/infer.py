import random
import glob

from torch.nn.functional import softmax
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize
import wandb

from config import *
from data import ImageDataset
from model import ResNet
from utils import *

def main():
    # load args
    args = load_infer_args()

    # load video
    clip_paths = glob.glob(os.path.join(RAW_DATA_PATH, "*"))
    video_path = os.path.join(random.choice(clip_paths), "video.mov")
    meta = load_metadata(video_path)
    duration = int(eval(meta['duration']))
    start_time = random.randint(0, duration - args.duration)

    # read video
    start_task(f"Reading {video_path}")
    video, _, _ = torchvision.io.read_video(video_path, start_pts=start_time, end_pts=start_time+args.duration, pts_unit="sec", output_format="TCHW")

    # load model
    start_task(f"Loading {args.model}:{args.version}")
    filepath = f"artifacts/{args.model}:{args.version}/{args.model}.pt"
    api = wandb.Api()
    artifact = api.artifact(f'mikasenghaas/bsc/{args.model}:{args.version}', type='model') # pyright: ignore
    artifact.download()

    model = ResNet(7)
    model.load_state_dict(torch.load(filepath))
    model.eval()

    # load data
    images = ImageDataset(PROCESSED_DATA_PATH, split="test")
    id2label = images.id2label

    # predict on random video clip
    fig, ax = plt.subplots()
    ax.set_title(f"{args.model}:{args.version}") # type: ignore
    ax.set_xticks([]) # type: ignore
    ax.set_yticks([]) # type: ignore

    img = ax.imshow(transforms.ToPILImage()(video[0])) # type: ignore

    """
    # transforms
    frame = resize(video[0], (224, 224)).float() # type: ignore
    frame = frame / 255.0
    frame = normalise_image(frame)

    show_image(frame, show=True, unnormalise=True)
    return
    """

    def animate(i):
        # transforms
        frame = resize(video[i], (224, 224)).float() # type: ignore
        frame = frame / 255.0
        frame = normalise_image(frame)

        logits = model(frame.unsqueeze(0))
        probs = softmax(logits, 1)
        prob, pred = torch.max(probs, 1)
        prob, pred = prob.item(), pred.item()

        print(f"Prediction: {id2label[pred]} (Confidence: {round(prob * 100, 2)}%)", end="\r")

        img.set_array(transforms.ToPILImage()(video[i])) # type: ignore

        return [img]

    a = animation.FuncAnimation(fig, animate, frames=len(video), interval=1, blit=True)
    plt.show()
    return

    for frame in video:
        print(frame.shape)
        break

        title = f"Pred: {id2label[pred]} ({round(prob * 100, 1)}%), True: {id2label[label]}"
        show_image(image, title=title, unnormalise=True, show=True)

if __name__ == "__main__":
    main()
