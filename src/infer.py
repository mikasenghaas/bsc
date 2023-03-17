import random
import glob

import cv2
from torch.nn.functional import softmax
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import wandb

from config import *
from data import ImageDataset
from model import FinetunedImageClassifier
from utils import *


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
        f'mikasenghaas/bsc/{args.model}:{args.version}', type='model')  # pyright: ignore
    filepath = os.path.join(BASEPATH, "artifacts",
                            f"{args.model}:{args.version}")
    if not os.path.exists(filepath):
        artifact.download(root=filepath)

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
        video_path = os.path.join(
            RAW_DATA_PATH, args.split, args.clip, "video.mov")

    if args.gradcam:
        if args.model == "resnet18":
            target_layers = [model.model.layer4[-1]]
        else:
            raise Exception(f"GradCam is not yet implemented for {args.model}")

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # set up video capture
    cap = cv2.VideoCapture(video_path)  # type: ignore

    start_task(f"Starting Inference on {'/'.join(video_path.split('/')[-3:])}")
    while True:
        # read next frame
        _, frame = cap.read()  # np.ndarray, (H=1920, W=1080, C=3), dtype=np.uint8
        frame_tensor = torch.tensor(frame).permute(
            2, 0, 1)  # torch.tensor, (C, H, W)
        # change channel to RGB from BGR
        frame_tensor = frame_tensor[[2, 1, 0], :, :]
        transformed_tensor = transform(
            frame_tensor).unsqueeze(0)  # transform tensor

        if frame_tensor == None:
            break

        # predict frame
        logits = model(transformed_tensor)
        probs = softmax(logits, 1)
        prob, pred = torch.max(probs, 1)
        prob, pred = prob.item(), pred.item()
        class_label = id2class[pred]

        # gradcam heatmap
        if args.gradcam:
            # type: ignore np.ndarray, (224, 224), dtype=np.uint8
            gradcam = cam(transformed_tensor).squeeze()
            gradcam = cv2.resize(
                gradcam, (frame.shape[1], frame.shape[0]))  # type: ignore

            # np.ndarray, (1920, 1080, 3), dtype=np.float32
            normalised_frame = frame.astype(np.float32) / 255.0

            frame = show_cam_on_image(
                normalised_frame, gradcam, image_weight=0.7)

        text = f"{class_label} ({round(100 * prob, 1)}%)"

        # overlay the prediction on the frame
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 1)  # type: ignore

        # display the frame with the prediction overlaid
        cv2.imshow(f"{args.model}:{args.version}", frame)  # type: ignore

        # exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # type: ignore
            break

    # release the video capture and close window
    cap.release()
    cv2.destroyAllWindows()  # type: ignore

    # end src/infer.py
    end_task("src/infer.py", start)


if __name__ == "__main__":
    main()
