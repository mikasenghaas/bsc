import random
import glob

import cv2
from torch.nn.functional import softmax
import torchvision
from torchvision import transforms
import wandb

from config import *
from data import ImageDataset
from model import FinetunedImageClassifier
from utils import *

def main():
    # start src/infer.py
    start = start_task("Running src/infer.py", get_timer=True)

    # load args
    args = load_infer_args()

    # load artifact
    start_task(f"Loading {args.model}:{args.version}")
    api = wandb.Api()
    artifact = api.artifact(f'mikasenghaas/bsc/{args.model}:{args.version}', type='model') # pyright: ignore
    if not os.path.exists(os.path.join(BASEPATH, "artifacts", f"{args.model}:{args.version}")):
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

    # get video path
    if args.clip is None:
        # random video path in split
        clip_paths = glob.glob(os.path.join(RAW_DATA_PATH, args.split, "*"))
        video_path = os.path.join(random.choice(clip_paths), "video.mov")
    else:
        # specific video path in split
        video_path = os.path.join(RAW_DATA_PATH, args.split, args.clip, "video.mov")

    # set up video capture
    cap = cv2.VideoCapture(video_path) # type: ignore

    start_task(f"Starting Inference on {'/'.join(video_path.split('/')[-3:])}")
    while True:
        # read next frame
        _, frame = cap.read()
        if type(frame) != np.ndarray:
            break

        frame_tensor = torch.tensor(frame).permute(2,0,1) # C, H, W
        frame_tensor = frame_tensor[[2,1,0], :, :] # change channel to RGB from BGR
        
        # predict frame
        logits = model(transform(frame_tensor).unsqueeze(0))
        probs = softmax(logits, 1)
        prob, pred = torch.max(probs, 1)
        prob, pred = prob.item(), pred.item()
        class_label = id2class[pred]
        
        text = f"{class_label} ({round(100 * prob, 1)}%)"
            
        # overlay the prediction on the frame
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1) # type: ignore
        
        # display the frame with the prediction overlaid
        cv2.imshow(f"{args.model}:{args.version}", frame) # type: ignore
        
        # exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # type: ignore
            break

    # release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows() # type: ignore

    # start src/infer.py
    end_task("src/infer.py", start)

if __name__ == "__main__":
    main()
