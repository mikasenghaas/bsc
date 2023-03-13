# BSc. DS - Mika Senghaas

This repository is under _active developement_.

## ⚙️ Setup

This project uses virtual environments (called `venv`) to easily replicate the Python environment originally used to produce the results. This includes a Python Version and the list of all Python packages (from PyPi) with their version requirements.

`pyenv-virtualenv` ([GitHub](https://github.com/pyenv/pyenv-virtualenv)) is used to manage both the Python version and external packages. It is a lightweight wrapper around the `pyenv` project ([Github](https://github.com/pyenv/pyenv)).

To setup the environment just run the bash script `setup` at the project root. It checks your path for the `pyenv` binaries and installs them from the official repository (using their automatic installation script). It then creates (if not yet present) the virtual environment and installs al dependencies. Re-running the script will not have any effect.

```bash
chmod +x setup && ./setup
```

To check that the environment is set up correctly, run the following list of commands.

```bash
pyenv version
```

This checks if `pyenv` is installed and the `bsc` virtual environment is installed.
The output should be `bsc (set by /User/<user>/<path>/bsc/.python-version)`.

```bash
python --version
```

The output should be `Python 3.10.X` (by default the latest stable release of Python `3.10` is installed).

Check whether all processed images are downloaded by verifying that the path `/src/data/processed` exists:

```bash
ls src/data/processed
```

Lastly, check if all package dependencies are installed.

```bash
pip list
```

It lists all packages and their versions. Make sure that you can find all packages listed in the `requirements.txt` file. Alternatively, you can run execute this inline Python expression. If the expressions runs without errors, you are ready to go!

```bash
python -c "import torch"
```

## 💾 Data and Preprocessing

All data within this project was gathered by myself over the period of a couple of weeks. A single sample `(x,y)` is a mapping between a high-dimensional sensory input data `x` and a location label `y`. The project experiments with the input data `x` being both single images (tensors of shape `(C, H, W)`) or sequences of images, aka. videos (tensor of shape `(F, C, H, W)`).

For the scope of this project, the chosen location is the [Institut for Medier, Erkendelse og Formidling](https://goo.gl/maps/e4v5dupcQgcbKuxS9), which is located at the Southern Campus of the Copenhagen University on Amagerbro (2300 Copenhagen). There are multiple reasons for why the location was found suitable:

- The space is **large** and **compartmentalised** (e.g. in floors), which allows to make the complexity of the location prediction incrementally more complex over time without having to switch location
- The location has many **characteristic features**, which are assumed to be learnable by the computer vision model
- The location is **located near me**, which allows for easy data gathering and extension of the dataset over multiple weeks

### Splits

To assess how well the model generalises to variying external factors, like weather, people crossing the frame, etc. the video clips are separated into a `train` and `test` split.

The `train` split contains video clips that capture the entire indoor space on a single day.

The `test` split contains video clips that were captured over multiple weeks to simulate naturally ocurring variance in the indoor space over time.

### Raw Video Clips and Annotation

A single raw data sample is a video clip captured while walking through the building. The height, angle and movement of the phone during recording tried to mimick how a user would use an app for inferring location. A raw video clip is typically in between 30-120s, leading to an uncompressed video size of up to `100MB`.

To allow for automatic processing of the raw videos, a specific directory structure and annotation method has to be used. All video clips are located in the folder `src/data/raw/train` or `src/data/raw/test`. A single raw video is identified by the year (`YY`), month (`MM`), day (`DD`) and video number (`VN`) on that specific day. The different features are combined into a video identifier of the general format `YYMMDD_VN`.

As an example, the first raw video recorded on March 13th 2023 the directory `230313_01` is created `YYMMDD_XX`). 

The video identifier serves as a name for the subdirectory within the base data path. This subdirectory is expected to contain _exactly_ two file:

- The raw **video clip**,  named `video.mov`
- A plaintext **annotation file**, named `annotations`, which annotates the video with corresponding location label. The formatting of the file matters for the automatic extraction to work. If the video clip shows location `Ground_Floor_Atrium` in between seconds 0 to 20 and after that `Ground_Floor_Red_Area` the file _must_ look like this:

   ```txt
   00:00,Ground_Floor_Atrium
   00:20,Ground_Floor_Red_Area
   ```

After all video clips located in the correct directory, correctly named, as well as annotated, your project should have this directory structure in the raw data path:

```bash
src/data/raw
├── test
│  ├── 230309_01
│  │  ├── annotations
│  │  └── video.mov
│  ├── ...
└── train
   ├── 230222_01
   │  ├── annotations
   │  └── video.mov
   ├── ...
```

### Extract Frames

To extract all frames, which are going to be fed into the model, one can run the `preprocess.py` script to automatically extract frames from the clips in the `train` or `test` split. Running the script from the root directory of this project with the `-h` flag gives

```bash
usage: preprocess.py [-h] [--split {train,test}] [--max-length MAX_LENGTH]
                     [--fps FPS]

options:
  -h, --help            show this help message and exit
  --split {train,test}  Which split to extract clips from
  --max-length 10       Maximum number of frames per extracted clip
  --fps 4               Number of frames per second (FPS) to extract
```

Thus, to preprocess all clips in the `train` and `test` split using default parameters, run the following commands:

```bash
python src/preprocess.py --split train
python src/preprocess.py --split test
```

All of this should generate a directory `src/data/preprocessed` with subdirectories `train` and `test`, each having subdirectories of all classes that were extracted. Each class directory contains a series of video clips (subclips from the raw video clips), which are identified through the same parameteres as the video (year (`YY`), month (`MM`), day (`DD`), video number (`VN`)) as well as a clip number (`CN`). This leads to a total clip identifier of `YYMMDD_VN_CN`, which serves as a directory names that contains all frames of that clip. Each frame has a unique number `FN` and is of format `.jpg`.

This leads to the following structure.

```bash
src/data/processed
├── test
│  ├── First_Floor_Corridor_1
│  │  ├── 230309_03_01
│  │  │  ├── 01.jpg
│  │  │  ├── ...
│  │  ├── ...
│  ├── ...
└── train
   ├── First_Floor_Corridor_1
   │  ├── 230302_01_04
   │  │  ├── 01.jpg
   │  │  ├── ...
   │  ├── ...
   ├── ...
```

If you're  

## Running the Project


## Notebooks
