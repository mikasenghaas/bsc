# Training and Deploying Computer Vision Models for Indoor Localisation on the Edge

⚠️ _This repository is under active development._

<img align="right" src="./assets/live-inference-gradcam.gif" height="400px"/>

Localisation is at the core of numerous localisation-based applications (LBAs). While digital solutions for outdoor positioning like GPS are wide-spread, similar systems in indoor spaces are rare and not unified in their approach. Most current solutions depend on additionally installed hardware in the indoor space, which is costly to install and maintain, inconvenient in the usage and often raises data privacy concerns. Therefore, this projects explores the possibilities of using machine learning (computer vision) to build a performant offline indoor positioning system.

The goal of this project is to train and deploy models that can predict where a user is located out of a fixed set of location labels for some indoor space. On the right you can see an example of the final result: Real-time inference on an unseen example video clip.

## 📱 Preview

You can try out a selection of trained models on your mobile phone! They are deployed using [PlayTorch](https://playtorch.dev) so that  . PlayTorch is a React Native (Typescript) bridge to the [PlayTorch Mobile SDK](https://pytorch.org/mobile/home/), which allows for rapid development of mobile demos. To try it out yourself follow these steps:

<img align="right" src="./assets/qr-180.png" height=150 width=150>

1. Download the PlayTorch App in the [App Store](https://apps.apple.com/us/app/playtorch/id1632121045) (iOS) or [Play Store](https://play.google.com/store/apps/details?id=dev.playtorch&hl=en&gl=US&pli=1) (Android)
2. Open the App and scan the QR code on the right
3. Go to the [Institut for Medier, Erkendelse og Formidling](https://goo.gl/maps/e4v5dupcQgcbKuxS9)

🔥 You are all set. Walk around the indoor space and observe the model's predictions.

## ⚙️ Setup

### Bootstrapping with Setup Script

The backbone of the project is written exclusively in Python. The quickest way to create a working Python environment to reproduce the results, train your own models etc. is to use the `setup` script. The script uses [`pyenv`](https://github.com/pyenv/pyenv) for managing the Python version and [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) to resolve a virtual environment.

```bash
chmod +x setup && ./setup
```

The script installs `pyenv`, `pyenv-virtualenv` and Python `3.10`. After that, it creates a virtual env called `bsc` with all dependencies installed and lastly extracts the processed training and test data.

### Custom Environment Tool

If you are using a different tool for handling Python versions and virtual environments, such as the natively shipped `python -m venv` or `conda`, and you want to use these, feel free to create a virtual environment using them. Then install all dependencies into:

```bash
pip install -r requirements.txt
```

To extract all data navigate into the directory `src/data` and unzip `processed.zip`.

```bash
cd src/data
unzip processed.zip && rm -rf processed.zip
```

### Troubleshooting

Check that your Python version is `3.10.X`:

```bash
python --version
```

Check that you have installed all relevant dependencies by running inline Python on a subset of the external packages:

```bash
python -c "import numpy; import torch; import torchvision"
```

Check whether all images are downloaded by verifying that the path `/src/data/processed` exists. The below command should not return an error code.

```bash
test -e src/data/processed
```

## 🚀 Running the Project

There is a number of different Python scripts and notebooks that enable you to produce results.

All model and data configurations that are used for evaluating the experiments mentioned in the final report are specified in the `train` script, which you can simply run using `./train` from the root.

If you want to train, infer or do other some yourself, you will have to run the corresponding scripts individually.

_Note, that all scripts must be run from the root of the project for paths to resolve properly._

### Training

The `train.py` script is the central script to train _different models_, on different _data configurations_ and with different hyperparameters. For example, to train ResNet18 on all data using default hyperparameters and logging to W&B:

```bash
python src/train.py -M resnet18 --all-classes --wandb-log
```

Find out more about all hyperparameters that you can tweak by running:

```txt
$ python src/train.py -h

usage: train.py [-h] -M MODEL [-V VERSION] [--pretrained | --no-pretrained] [--filepath FILEPATH]
                [--include-classes INCLUDE_CLASSES [INCLUDE_CLASSES ...]] [--all-classes | --no-all-classes]
                [--ground-floor | --no-ground-floor] [--first-floor | --no-first-floor] [--ratio RATIO]
                [--device {cpu,cuda,mps}] [--wandb-log | --no-wandb-log] [--wandb-name WANDB_NAME]
                [--wandb-group WANDB_GROUP] [--wandb-tags WANDB_TAGS [WANDB_TAGS ...]] [--max-epochs MAX_EPOCHS]
                [--batch-size BATCH_SIZE] [--lr LR] [--step-size STEP_SIZE] [--gamma GAMMA]
```

_For more detailed output run the command yourself._

### Inference

The `infer.py` script loads a random video clip from a specified data split and runs live predictions of a trained model that was logged to W&B, which are overlayed and displayed as a video instance.

For example, to predict on a random video clip from the test split using Version `v0` of the Resnet18 model class, run:

```bash
python src/infer.py -M resnet18 -V v0 --test
```

You can overlay a heatmap generated by the [GradCam]() algorithm to display the relevance of image regions for the final predictions by activating the `--gradcam` flag. Run the `-h` flag to see a summary of all parameters that can be specified.

```txt
$ python src/infer.py -h

usage: infer.py [-h] -M MODEL [-V VERSION] [--pretrained | --no-pretrained] [--device {cpu,cuda,mps}]
                [--gradcam | --no-gradcam] [--split {train,val,test}] [--clip CLIP]
```

_For more detailed output run the command yourself._

### Notebooks

There is a number of Jupyter Notebooks in the directory `notebooks`. Each should be self-explanatory when following them block-by-block, so I will just list what each of the includes here:

- `eda.ipynb` contains some verifications and basic data analysis on the gathered dataset
- `train.ipynb` shows how to fine-tune an image or video classifier for indoor localisation
- `evaluate.ipynb` contains a series of evaluation techniques for trained models that have been logged to W&B
- `optimise.ipynb` optimises a trained PyTorch model for mobile deployment
