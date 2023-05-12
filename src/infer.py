import glob
import os
import random

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.nn.functional import softmax
from torchvision import transforms

import wandb
from config import BASEPATH, RAW_DATA_PATH, WANDB_PROJECT
from utils import load_infer_args, start_task, end_task

from modules import MODULES
from defaults import DEFAULT


def main():
    """
    Main function for src/infer.py
    """
    # start src/infer.py
    start = start_task("Running src/infer.py", get_timer=True)

    # load args
    args = load_infer_args()

    # load artifact
    start_task(f"Loading {args.model}:{args.version}")
    api = wandb.Api()
    artifact = api.artifact(
        f"mikasenghaas/{WANDB_PROJECT}/{args.model}:{args.version}", type="model"
    )  # pyright: ignore
    filepath = os.path.join(BASEPATH, "artifacts", f"{args.model}:{args.version}")
    if not os.path.exists(filepath):
        artifact.download(root=filepath)

    # get default configs
    config = DEFAULT[args.model]
    model_type = config["general"]["type"]

    # get modules
    TRANSFORM = MODULES[model_type]["transform"]
    DATA = MODULES[model_type]["data"]
    MODEL = MODULES[model_type]["model"]

    # initialise modules with default config
    transform = TRANSFORM(**config["transform"])
    data = DATA(**config["dataset"], split="test", transform=transform)
    model = MODEL(**config["model"])

    # load id2class
    id2class = data.id2class

    # load model weights
    model_path = f"artifacts/{args.model}:{args.version}/{args.model}.pt"
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

    if args.gradcam:
        match args.model:
            case "resnet18":
                target_layers = [model.model.layer4[-1]]
            case _:
                raise NotImplementedError(
                    f"GradCAM not implemented for this {args.model}"
                )

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # set up video capture
    cap = cv2.VideoCapture(video_path)  # type: ignore

    start_task(f"Starting Inference on {'/'.join(video_path.split('/')[-3:])}")
    if model_type == "video":
        num_frames = config["transform"]["num_frames"]
        sampling_rate = config["transform"]["sampling_rate"]
        frame_history = []

    frame_index = 0
    logits = None
    while True:
        # read next frame
        _, frame = cap.read()

        # break if no frame
        if frame is None:
            break

        # preprocess frame
        frame_tensor = torch.tensor(frame).permute(2, 0, 1)  # torch.tensor, (C, H, W)
        frame_tensor = frame_tensor[[2, 1, 0], :, :]  # change channel to RGB from BGR

        if model_type == "video":
            if frame_index % sampling_rate == 0:  # 3 fps
                frame_tensor = frame_tensor.unsqueeze(0) # add temporal dim
                frame_tensor = frame_tensor.permute(1, 0, 2, 3) # C, T, H, W
                # frame_tensor = frame_tensor.unsqueeze(0) # add batch dim
                frame_history.append(frame_tensor)

        # predict frame
        match model_type:
            case "image":
                frame_tensor = transform(transforms.ToPILImage()(frame_tensor))
                frame_tensor = frame_tensor.unsqueeze(0)  # T, C, H, W

                logits = model(frame_tensor)
            case "video":
                if len(frame_history) == num_frames:
                    frame_tensors = torch.cat(frame_history, dim=1)
                    sample = transform({"video": frame_tensors})
                    frame_tensors = sample["video"]
                    logits = model(frame_tensors.unsqueeze(0))
                    frame_history = []
            case _:
                raise NotImplementedError(f"Model type {model_type} not implemented")

        if logits is not None:
            probs = softmax(logits, 1)
            prob, pred = torch.max(probs, 1)
            prob, pred = prob.item(), pred.item()
            class_label = id2class[pred]

            text = f"{class_label} ({round(100 * prob, 1)}%)"
        else:
            text = "No prediction yet."

        # gradcam heatmap
        if args.gradcam:
            # type: ignore np.ndarray, (224, 224), dtype=np.uint8
            gradcam = cam(frame_tensor).squeeze()
            gradcam = cv2.resize(
                gradcam, (frame.shape[1], frame.shape[0])
            )  # type: ignore

            # np.ndarray, (1920, 1080, 3), dtype=np.float32
            normalised_frame = frame.astype(np.float32) / 255.0

            frame = show_cam_on_image(normalised_frame, gradcam, image_weight=0.7)

        # overlay the prediction on the frame
        cv2.putText(
            frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 1
        )  # type: ignore

        # display the frame with the prediction overlaid
        cv2.imshow(f"{args.model}:{args.version}", frame)  # type: ignore

        # exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):  # type: ignore
            break

        frame_index += 1

    # release the video capture and close window
    cap.release()
    cv2.destroyAllWindows()  # type: ignore

    # end src/infer.py
    end_task("src/infer.py", start)


if __name__ == "__main__":
    main()
