# MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label

[![Paper](https://img.shields.io/badge/arXiv-2604.01646-b31b1b.svg)](https://arxiv.org/pdf/2604.01646)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://visualaikhu.github.io/MonoSAOD/)

The paper has been accepted by CVPR 2026.

Official implementation of **MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label**

## Introduction
 
![MonoSAOD](main_fig.png)

## Introduction

MonoSAOD is the first framework to address sparsely-annotated monocular 3D object detection. 
Unlike existing methods that assume fully labeled 3D annotations, we explicitly tackle the realistic setting where only a fraction of visible objects are annotated.

We introduce a geometry-consistent augmentation module (RAPA) and a prototype-guided pseudo-label filtering module (PBF). 
RAPA leverages sparse ground truths by placing clean object patches onto valid road regions while preserving 3D geometric consistency. 
PBF selects reliable pseudo-labels by jointly evaluating feature-level prototype similarity and depth uncertainty, preventing erroneous 3D supervision.

By combining geometry-aware augmentation and uncertainty-aware pseudo-labeling within a teacher–student framework, MonoSAOD enables robust 3D learning under severe annotation sparsity.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+
- Other dependencies as specified in requirements (if available)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/VisualAIKHU/MonoSAOD.git
cd MonoSAOD
```

2. Create a virtual environment:
```bash
conda create -n monosaod python=3.10
conda activate monosaod
```

3. Setup:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # install this according to your nvcc version
pip install -r requirements.txt

cd lib/models/monodetr/ops/
bash make.sh

cd ../../../..

mkdir logs
```
## Dataset

### KITTI 3D Object Detection Dataset

This project uses the KITTI 3D object detection dataset with patch augmentation. To prepare the dataset:

1. Download KITTI dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php)

2. Extract and organize the dataset:
```
KITTI/object/
├── ImageSets/
    ├── train.txt
    ├── val.txt
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── road_masks/             # Road masks for patch augmentation
│   ├── 30/                     # sparsity 
│   │   ├── patch/              # Patch images
│   │   └── label_patch/        # Patch labels
└── testing/
    ├── calib/
    ├── image_2/
```

3. Update the dataset path in `configs/monodetr.yaml`:
```yaml
dataset:
  root_dir: '/path/to/KITTI/object/'
  patch_dir: '/path/to/KITTI/object/training/30/patch'
  patch_label_dir: '/path/to/KITTI/object/training/30/label_patch'
  mask_dir: '/path/to/KITTI/object/training/patch_dirs/road_masks'
```

### Patch Augmentation Setup

Download the patch files and road masks from here: [patch](https://drive.google.com/file/d/1rO1oI8lneJBAw8cz_W2ByCJ2LTSPI3CY/view?usp=sharing), [road masks](https://drive.google.com/file/d/1f_Y_C2QX_29zFQJ7XKpGlA3dn8rmDxni/view?usp=sharing)

 **patch files:**
   - **`patch/`**: Contains cropped object patches from training images
   - **`label_patch/`**: Contains corresponding labels for patches
   - **`road_masks/`**: Contains binary masks of road regions for augmentation

## Training

### Basic Training

To start training with the default configuration:

```bash
bash train.sh configs/monodetr.yaml > logs/monodetr.log
```

## Evaluation

### Evaluate on Validation Set

To evaluate the model:

```bash
bash test.sh configs/monodetr.yaml
```

## Directory Structure

```
MonoSAOD/
├── configs/              # Configuration files
├── lib/
│   ├── datasets/        # Dataset loaders
│   │   └── kitti/       # KITTI-specific code
│   ├── helpers/         # Training helpers
│   ├── losses/          # Loss functions
│   └── models/          # Model definitions
│       └── monodetr/    # MonoDETR model
├── tools/
│   └── train_val.py     # Training and evaluation script
├── train.sh             # Training script
└── README.md            # This file
```

## License

This project is based on MonoDETR. Please refer to the LICENSE file for more information.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{jung2026monosaod,
  title={MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label},
  author={Jung, Junyoung and Kim, Seokwon and Kim, Jung Uk},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4718--4727},
  year={2026}
}
```
