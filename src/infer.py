import random
import glob

from torch.nn.functional import softmax
import torchvision
from torchvision import transforms
import wandb

from config import *
from data import ImageDataset
from model import FinetunedImageClassifier
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

    # load artifact
    start_task(f"Loading {args.model}:{args.version}")
    api = wandb.Api()
    artifact = api.artifact(f'mikasenghaas/bsc/{args.model}:{args.version}', type='model') # pyright: ignore
    artifact.download()

    # load data
    config = ImageDataset.default_config()
    images = ImageDataset(**config)
    id2class = images.id2class

    # paths to model, transforms and config file
    model_path = f"artifacts/{args.model}:{args.version}/{args.model}.pt"
    config_path = f"artifacts/{args.model}:{args.version}/config.json"
    transforms_path = f"artifacts/{args.model}:{args.version}/transforms.pkl"

    # load transforms
    transform = load_pickle(transforms_path)

    # load model
    config = load_json(config_path)
    model = FinetunedImageClassifier(**config)
    model.load_state_dict(torch.load(model_path))

    # set eval mode
    model.eval()

    # predict on random video clip
    fig, ax = plt.subplots()
    ax.set_title(f"{args.model}:{args.version}") # type: ignore
    ax.set_xticks([]) # type: ignore
    ax.set_yticks([]) # type: ignore

    img = ax.imshow(transforms.ToPILImage()(video[0])) # type: ignore

    def animate(i):
        # transforms
        logits = model(transform(video[i]).unsqueeze(0))
        probs = softmax(logits, 1)
        prob, pred = torch.max(probs, 1)
        prob, pred = prob.item(), pred.item()

        print(f"Prediction: {id2class[pred]} (Confidence: {round(prob * 100, 2)}%)", end="\r")

        img.set_array(transforms.ToPILImage()(video[i])) # type: ignore

        return [img]

    a = animation.FuncAnimation(fig, animate, frames=len(video), interval=1, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
