# Training and Deploying Computer Vision Models for Indoor Localisation on the Edge

<img align="right" src="./assets/live-inference-gradcam.gif" height="400px"/>

This repository contains the code and final report of my bachelor thesis at the
IT University of Copenhagen. The thesis was supervised by [Stella
Grasshof](https://pure.itu.dk/en/persons/stella-grasshof). 

The goal of this project was to train and deploy single-frame and video
classification models to provide accurate indoor localisation predictions with a
room-level granularity. The project entailed the collection and annotation of a
novel video dataset tailored for indoor localisation and the rigorous training
and evaluation of various modern neural network architectures.

An example of a live inference of a trained model can be seen on the right. The
model is trained to predict the room of a video clip. The model is deployed on
the edge using [PlayTorch](https://playtorch.dev), a React Native bridge to the
[PlayTorch Mobile SDK](https://pytorch.org/mobile/home/).


## 📝 Abstract

In an increasingly urbanised and digitalised world, indoor localisation is
becoming a necessity for a wide variety of applications, ranging from personal
navigation to augmented reality. However, despite extensive research efforts,
indoor localisation remains a challenging task and no single solution is widely
adopted. Motivated by the success of deep learning in numerous computer vision
tasks, this study explores the feasibility of deep learning for accurate
room-level localisation in indoor spaces. Various neural network architectures
are trained and evaluated on a novel video dataset tailored for indoor
localisation. The findings reveal that deep learning approaches can provide
reasonable localisation results, even when trained on a small dataset. The
approach is currently limited by its inability to distinguish between visually
similar and adjacent areas, as well as biases within the training data. Despite
these shortcomings, the results are encouraging and inspire optimism about the
method’s practical viability.

## 📱 Preview

You can try out a selection of trained models on your mobile phone! They are
deployed using [PlayTorch](https://playtorch.dev). To try it out yourself follow
these steps:

<img align="right" src="./assets/qr-180.png" height=150 width=150>

1. Download the PlayTorch App in the
   [App Store](https://apps.apple.com/us/app/playtorch/id1632121045) (iOS) or
   [Play Store](https://play.google.com/store/apps/details?id=dev.playtorch&hl=en&gl=US&pli=1)
   (Android)
2. Open the App and scan the QR code on the right
3. Go to the [Institut for Medier, Erkendelse og
   Formidling](https://goo.gl/maps/e4v5dupcQgcbKuxS9)

🔥 You are all set. Walk around the indoor space and observe the model's
predictions.

## ⚙️ Setup

### Python Version

The backbone of this project is written in Python. The project runs in any minor
version of Python `3.10`. Make sure that you have the correct Python version by
running `python --version`. If you are using a different version, you can use
[`pyenv`](https://github.com/pyenv/pyenv) to install the correct version.

### Dependencies

All dependencies are managed  with [`Poetry`](https://python-poetry.org/).
Assuming that you have `poetry` installed, you can install all dependencies by
running:

```bash
poetry install
```

This command will create a virtual environment for you and install all relevant
dependencies into it. You can activate the virtual environment by running:

```bash
poetry shell
```

Alternatively you can run all commands from your regular shell session by
prefixing the command with `poetry run`, e.g. to run the training python script,
you would type:

```bash
poetry run src/train.py ...
```

If you wish to use another dependency manager, you can find a list of all
dependencies in `pyproject.toml`.

### Data

Because of the large data size, this repository **does not contain the raw data**.
Instead if contains a zip file with the processed frames and videos that can be
used to train the models.

Before running the project, you will have to extract the data. To extract all
data navigate into the directory `src/data` and unzip `images.zip` and
`videos.zip`

```bash
cd data
unzip images.zip && rm -rf images.zip
unzip videos.zip && rm -rf videos.zip
```

_Note, that you need to extract the data before running all scripts in the data,
because the data class depends on the data being extracted locally._

## 🚀 Running the Project

The project offers two main entry points for running the project:

1. `train.py`: Train a single model with chosen training hyperparameters
1. `eval.py`: Evaluate a model's performance and efficiency on test split
2. `infer.py`: Run live inference on a exemplary video clip

The job files `train.job` and `eval.job` are used to run the experiments on the
[SLURM](https://slurm.schedmd.com/documentation.html) cluster of HPC of the IT
University of Copenhagen.

### Training

The `train.py` script is the central script to train _different models_ and with
different hyperparameters. For example, to train ResNet18 on all data using
default hyperparameters and logging to W&B:

```bash
python src/train.py -M resnet18
```

You can see the identifiers for all models within this project in the file
`defaults.py`.  Find out more about all hyperparameters that you can tweak by running:

```txt
$ python src/train.py -h

usage: train.py [-h] -M MODEL [-V VERSION] [--wandb-log | --no-wandb-log] [--wandb-name WANDB_NAME]
                [--wandb-group WANDB_GROUP] [--wandb-tags WANDB_TAGS [WANDB_TAGS ...]]
                [--epochs EPOCHS] [--device DEVICE] [--batch-size BATCH_SIZE] [--lr LR]
```

_For more detailed output run the command yourself._

### Evaluation

The `eval.py` script loads a trained model, as specified by the model identifier
and version number, from the public W&B repository. It then evaluates the model
on the test split and logs the results to W&B.

Unless you have trained a model yourself, you do not need to run this script.

### Inference

The `infer.py` script loads a trained model, as specified by the model
identifier and version number, from the public W&B repository. It then selects a
random or specified video clip from the test split and runs live predictions of
the model on the video clip. The top prediction and confidence score are
overlayed and displayed as a video instance.

Because of data size limitations on GitHub, only a single video clip is public
in this repository (see `data/raw/230313_04/video.mov`). To run inference on this
video clip for `v0` of ResNet18, run:

```bash
python src/infer.py -M resnet18 -V v0 --clip 230313_04
```

These are the available arguments to the script:

```txt
$ python src/infer.py -h

usage: infer.py [-h] -M MODEL [-V VERSION] [--gradcam | --no-gradcam] 
                     [--split {train,test}] [--clip CLIP]
```

_For more detailed output run the command yourself. Further, note that the
gradcam overlay is only available for the ResNet18 model._

### Notebooks

There is a number of Jupyter Notebooks in the directory `notebooks`, which were
used to gather statistics, results and generate visualisations for the final
report. Each should be self-explanatory when following them block-by-block, so
this is only a short list of the included notebooks:

- `eda.ipynb` contains some verifications and basic data analysis on the
  gathered dataset
- `optimise-mobile.ipynb` contains the process of optimising a model for mobile
  deployment
- `results.ipynb` contains a series of evaluation techniques for trained models
  that have been logged to W&B
