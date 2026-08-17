# MIC-OPCC: Multi-Indexed Convolution model for Octree Point Cloud Compression

Multi-Indexed Convolution introduces an alternative approach to spatial feature extraction, which we use in our entropy model to compress the occupation symbols of octree-encoded point clouds. This method offers reduced time and memory usage per point compared to related work.

## Requirements

This project is written in Python using Tensorflow and Keras.
It is expected to be run on a Linux OS using a NVIDIA GeForce RTX3090 or higher.
It is designed and tested on the following dependencies: 

```r
# Required
python:                 3.10
tensorflow:             2.9.0
keras:                  2.9.0
numba:                  0.60
scipy:                  1.12.0

# Optional
tensorflow-probability: 0.15.0 # for compression via tfc
tensorflow-compression: 2.9.2  # for compression via tfc
python-pcl==0.3.0rc1           # for loading .ply files
```

## Installation

This project can be run locally or in a docker container with GPU support.

After you've checked out this project via:

```bash
git clone https://github.com/bugerry87/mic-opcc.git
```

You may wish to change permissions on all shell scripts to make them executable, by:

```bash
cd ./mic-opcc
chmod +x *.sh
```

### Local Installation

For running this project locally, we recommend using [Anaconda](https://www.anaconda.com/download) or similar to set up the basic environment.

```bash
conda create -n mic-opcc -c nvidia -c conda-forge -y \
  python=3.10 \
  cudatoolkit=11.2.0 \
  cudnn=8.9.2.26 \
  pcl=1.13.0 \
  python-pcl=0.3.0rc1
```

We recommeded to install the following remaining dependencies via `pip`:

```bash
conda activate mic-opcc
pip install tensorflow==2.9.0 \
  tensorflow-probability==0.15.0 \
  tensorflow-compression==2.9.2 \
  keras==2.9.0 \
  numba==0.60 \
  scipy==1.12
```

Last but not least, we have to add conda's `./lib` to the environment variable `LD_LIBRARY_PATH`.
This can be done via: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib`.
We recommend to add this statement to `$CONDA_PREFIX/etc/conda/activate.d/libglib_activate.sh` so it applies automatically when activating `mic-opcc`.

Either, use this command:

```bash
conda activate mic-opcc
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"' >> "$CONDA_PREFIX/etc/conda/activate.d/libglib_activate.sh"
```

Or detect that file and modify it manually.
The script `$CONDA_PREFIX/etc/conda/activate.d/libglib_activate.sh` should finally look like this:

```bash
export GSETTINGS_SCHEMA_DIR_CONDA_BACKUP="${GSETTINGS_SCHEMA_DIR:-}"
export GSETTINGS_SCHEMA_DIR="$CONDA_PREFIX/share/glib-2.0/schemas"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

### Docker Build

Having a docker host with GPU support, build an image by using the following script:

```bash
sh ./buildimage.sh
```

This will build an image based on tensorflow `v2.9.0`, having all dependencies installed and a with direct entry point on `run_mic_pcc.py`. 

## Datasets

This project was designed and tested for the following 3 datasets:

- [Semantic KITTI](https://semantic-kitti.org/) is an 80GB record of sequential LiDAR scans from a sensor mounted on top of a car's roof.
- [8iVFBv2](http://plenodb.jpeg.org/pc/8ilabs/) is a set of 5.5GB voxelized full human bodies to a precision of 10 bits per dimension.
- [MVUB](http://plenodb.jpeg.org/pc/microsoft/) is a set of about 6.4GB voxelized busts to a precision of 10 bits per dimension.

### Semantic KITTI

Download and extract the [Semantic KITTI](https://semantic-kitti.org/) dataset to `./data/semantic-kitti/`.

```bash
sh download_semantic_kitti.sh ./data/semantic-kitti/
```

**Note:** You may choose another destination path, but then mind to adjust the paths in `./samples/kitti_train_index.txt` and `./samples/kitti_test_index.txt`.

### MPEG's 8iVFBv2

Download and extract MPEG's [8iVFBv2](http://plenodb.jpeg.org/pc/8ilabs/) dataset to `./data/mpeg/8iVFBv2/`.

```bash
sh download_mpeg_8iVFBv2.sh ./data/mpeg/8iVFBv2/
```

**Note:** You may choose another destination path, but then mind to adjust the paths in `./samples/mpeg_train_index.txt` and `./samples/mpeg_test_index.txt`.

### MVUB

Download and extract Microsoft's Voxelized Upper Bodies [MVUB](http://plenodb.jpeg.org/pc/microsoft/) dataset to `./data/mvub/`.

```bash
sh download_mvub.sh ./data/mvub/
```

**Note:** You may choose another destination path, but then mind to adjust the paths in `./samples/mvub_train_index.txt` and `./samples/mvub_test_index.txt`.

## Usage

```
MultiIndexedBeamConvolutionPCC

options:
  -h, --help            show this help message and exit
  --name STR            Name of the model (default=mic-opcc)
  --train_index [PATH ...], -X [PATH ...]
                        A index file to training data
  --val_index [PATH ...], -Y [PATH ...]
                        A index file to validation data
  --test_index [PATH ...], -T [PATH ...]
                        A index file to test data
  --xshape SHAPE [SHAPE ...]
                        Shape of the input data
  --xtype TYPE          Type of the input data
  --xformat FORMAT      Format of the input data
  --offset Float [Float ...]
                        Quantization offset
  --scale Float [Float ...]
                        Quantization scale
  --epochs INT, -e INT  Num of epochs
  --learning_rate Float
                        Learning rate for the Adam optimizer (default=1e-4)
  --learning_decay Float
                        Learning rate for the Adam optimizer (default=0.9)
  --monitor STR         Choose the metric to be monitored for checkpoints and early stopping (default=automatic)
  --save_best_only      Whether to save only best model or (default) not
  --stop_patience INT   The early stopping patience (deactivate = -1)
  --steps_per_epoch INT
                        Define to train on a subset
  --validation_freq INT
                        Validation frequency
  --validation_steps INT
                        Define to validate on a subset
  --test_freq INT       Test frequency (default=1)
  --test_steps INT      Define for test on a subset
  --test_precision INT  Define precision during test
  --range_coder STR     Select range coder implementation
  --shuffle INT         Size of the shuffle buffer
  --precision INT, -P INT
                        Quantization precision
  --tree_type INT, -t INT
                        Tree type: 1 = binary tree, 2 = quatree, 3 = octree
  --qmode STR, -q STR   Quantization precision
  --derotate            Sort axis by major components
  --eroder              Run in eroder mode
  --rotate STR          Random rotation augmentation - use "xyz" (default="")
  --grouping STR, -g STR
                        Grouping strategy
  --slices INT [INT ...], -S INT [INT ...]
                        Tree slices
  --chunk INT, -C INT   Chunk level
  --kernels INT [INT ...], -k INT [INT ...]
                        num of kernel units
  --windows INT [INT ...], -w INT [INT ...]
                        window size
  --embedding INT, -E INT
                        num of embedding units
  --convolutions INT [INT ...], -c INT [INT ...]
                        number of convolution layers
  --head_size INT [INT ...], -n INT [INT ...]
                        the dense layer size after convolution
  --salt FLOAT          Ratio to add salt to data - adds random points (default=0.0)
  --pepper FLOAT        Ratio to add pepper to data - removes random points (default=0.0)
  --dropout FLOAT       Dropout (default=0.0)
  --seed INT            Initial model seed
  --log_dir PATH        Model type (default=logs)
  --verbose INT, -v INT
                        verbose level (see tensorflow)
  --profiler INT        Activate profiler per batch (default=0)
  --cpu                 Whether to allow cpu or (default) force gpu execution
  --checkpoint PATH     Load from checkpoint
  --generate FLOAT      Generate a confidence point cloud at the end (default=0.0)
```

## Run a Toy Example

Run a Toy Example on `mini_index.txt` that contains one single point cloud sample and test the compression result on the very same sample.
This command starts a training session of 1 epochs with 10 iterations each at a quantization precision of 12bits per dimension.
Each step process one sub-group of 8 sub-groups per octree layer.
Hance, the script will conclude with 100x12x8 = 9600 iterations.
Each iteration end with a test session, evaluating the compression and accuracy.
The model is configured to 3 sub-modules.
Each sub-module is dedicated to 4 octree layers:

- Sub-module 1 applies 4 convolutions and 1 fully-connected layer.
- Sub-module 2 applies 8 convolutions and 2 fully-connected layers.
- Sub-module 3 applies 12 convolutions and 3 fully-connected layers.

This session allows to be run on CPU and uses an Arithmetic Range Coder implementation based on Numba.

We provide several methods to run this project:

**Run locally by direct command:**

```bash
python ./run_mic_pcc.py -X ./samples/mini_index.txt -T ./samples/mini_index.txt -P 12 -e 1 --steps_per_epoch 10 -S 0 4 8 12 -c 4 8 12 -n 1 2 3 --range_coder=nrc --cpu
```

Or use one of our helper scripts that already pre-define a certain set of hyper-parameters.
Any parameters can be overrided by simply tailing them behind the script.

**Run locally by helper script:**

```bash
sh ./train_mic_mini.sh "[override parameters here]"
```

To run the same process in a docker container, set the environment variable `DOCKER=1` as prefix.

**Run in Docker by helper script:**

```bash
DOCKER=1 sh ./train_mic_mini.sh "[override parameters here]"
```

## Run Proposed Models

For each epoch the model stores an `.h5` file of weights as a checkpoint to the default path in `./logs`.
The checkpoint doesn't store any hyper-parameters.
One has to reproduce all hyper-parameters in the command line to restore the exact same model.
The checkpoints gets rejected if its weights do not fit into the generated model.
Please check the console output for warnings in this matter.

### Run MIC-OPCC on Semantic KITTI from Checkpoint

To continue a training session from checkpoint, simple load the weights to `--checkpoint`:

```bash
DOCKER=1 sh train_mic_kitti.sh --checkpoint ./logs/mic-opcc-kitti/ckpts/kitti_0007-0.343.weights.h5
```

You may download and extract a pre-trained model from [here](https://drive.google.com/file/d/1bdlrwOwI51HyfVMTxBG5CpPeFJAFxGOo/view?usp=drive_link).
In this example, we store the checkpoint to our default path `./logs`.
That checkpoint match with the pre-defined hyper-parameters in `train_mic_kitti.sh`.

To run a test session only, just nullify the training samples in `-X`: 

```bash
DOCKER=1 sh train_mic_kitti.sh -X "" --checkpoint ./logs/mic-opcc-kitti/ckpts/kitti_0007-0.343.weights.h5
```

### Run MIC-OPCC on MPEG's 8iVFBv2 from Checkpoint

To continue a training session from checkpoint, simple load the weights to `--checkpoint`:

```bash
DOCKER=1 sh train_mic_mpeg.sh --checkpoint ./logs/mic-opcc-mpeg/ckpts/mpeg_0009-0.144.weights.h5
```

You may download and extract a pre-trained model from [here](https://drive.google.com/file/d/1Hms26PvDc3f2HaR93xDylCwFKErgLyms/view?usp=drive_link).
In this example, we store the checkpoint to our default path `./logs`.
That checkpoint match with the pre-defined hyper-parameters in `train_mic_mpeg.sh`.

To run a test session only, just nullify the training samples in `-X`: 

```bash
DOCKER=1 sh train_mic_mpeg.sh -X "" --checkpoint ./logs/mic-opcc-mpeg/ckpts/mpeg_0009-0.144.weights.h5
```

### Run MIC-OPCC on Microsoft's MVUB from Checkpoint

To continue a training session from checkpoint, simple load the weights to `--checkpoint`:

```bash
DOCKER=1 sh train_mic_mvub.sh --checkpoint ./logs/mic-opcc-mpeg/ckpts/mvub_0009-0.144.weights.h5
```

You may download and extract a pre-trained model from [here](https://drive.google.com/file/d/16IXjyA9If7ChkpoiE9TcxmXK0V6Pppr6/view?usp=drive_link).
In this example, we store the checkpoint to our default path `./logs`.
That checkpoint match with the pre-defined hyper-parameters in `train_mic_mvub.sh`.

To run a test session only, just nullify the training samples in `-X`: 

```bash
DOCKER=1 sh train_mic_mvub.sh -X "" --checkpoint ./logs/mic-opcc-mvub/ckpts/mpeg_0009-0.144.weights.h5
```

## Trouble Shooting

- Make sure, the dataset is downloaded and extracted.
- If you changed the location of the dataset, please adjust the `_index.txt` files accordingly.
- Using different hyper-parameters, requires new training from scratch.

## TODO's

- Currently, there is no standalone decoder.
  This model requires an arithmetic coder that emits the current bit position of the read file while decoding,
  such that the model can proceed on the current data stream while estimating the probabilities for the next group.
- Our Numba based range coder (`nrc`) is currently broken. Use `tfc`.

## Citation

If our work is related to yours, please consider to cite our following publications:

### MIC-OPCC v1.0

Baulig, G., Guo, JI. (2026). Methodology and Results of MIC-OPCC: Multi-Indexed Convolution Model for Octree Point Cloud Compression. In: Li, P., et al. Advances in Computer Graphics. CGI 2025. Lecture Notes in Computer Science, vol 16508. Springer, Cham. https://doi.org/10.1007/978-3-032-22264-0_35

```bibtex
@InProceedings{10.1007/978-3-032-22264-0_35,
author="Baulig, Gerald
and Guo, Jiun-In",
editor="Li, Ping
and Ma, Lizhuang
and Wan, Liang
and Sheng, Bin
and Kim, Jinman
and Thalmann, Daniel
and Magnenat-Thalmann, Nadia",
title="Methodology and Results of MIC-OPCC: Multi-Indexed Convolution Model for Octree Point Cloud Compression",
booktitle="Advances in Computer Graphics",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="443--454",
abstract="For point cloud compression, capturing sufficient spatial context is crucial for accurately modeling the point cloud distribution. However, voxel-based methods tend to lose effectiveness when dealing with sparse point clouds of higher precision, as the context they gather becomes less comprehensive. This study introduces an octree-based point cloud compression method that utilizes an entropy model powered by deep learning to estimate probabilities, which are then used to guide an Arithmetic Range Coder in reducing the bit rate of the serialized octree code. Our proposed model extracts local features using lightweight 1D convolution applied in varied ordering and analyzes causal relationships by optimizing the cross-entropy. This approach efficiently replaces the voxel-convolution techniques and attention models used in previous works, providing significant improvements in both time and memory consumption. The effectiveness of our model is demonstrated on three datasets, where it outperforms recent deep learning-based compression models in this field. This project can be found at https://github.com/bugerry87/mic-opcc.",
isbn="978-3-032-22264-0"
}
```

### MIC-OPCC v2.0

Baulig, G.; Guo, J.-I. Autoregressive and Residual Index Convolution Model for Point Cloud Geometry Compression. Sensors 2026, 26, 1287. https://doi.org/10.3390/s26041287

```bibtex
@Article{s26041287,
AUTHOR = {Baulig, Gerald and Guo, Jiun-In},
TITLE = {Autoregressive and Residual Index Convolution Model for Point Cloud Geometry Compression},
JOURNAL = {Sensors},
VOLUME = {26},
YEAR = {2026},
NUMBER = {4},
ARTICLE-NUMBER = {1287},
URL = {https://www.mdpi.com/1424-8220/26/4/1287},
ISSN = {1424-8220},
ABSTRACT = {This study introduces a hybrid point cloud compression method that transfers from octree-nodes to voxel occupancy estimation to find its lower-bound bitrate by using a Binary Arithmetic Range Coder. In previous attempts, we demonstrated that our entropy compression model based on index convolution achieves promising performance while maintaining low complexity. However, our previous model lacks an autoregressive approach, which is apparently indispensable to compete with the current state-of-the-art of compression performance. Therefore, we adapt an autoregressive grouping method that iteratively populates, explores, and estimates the occupancy of 1-bit voxel candidates in a more discrete fashion. Furthermore, we refactored our backbone architecture by adding a distiller layer on each convolution, forcing every hidden feature to contribute to the final output. Our proposed model extracts local features using lightweight 1D convolution applied in varied ordering and analyzes causal relationships by optimizing the cross-entropy. This approach efficiently replaces the voxel convolution techniques and attention models used in previous works, providing significant improvements in both time and memory consumption. The effectiveness of our model is demonstrated on three datasets, where it outperforms recent deep learning-based compression models in this field.},
DOI = {10.3390/s26041287}
}
```

## License

This code is provided by myself for purely non-commercial, research purposes. It may not be used commercially in a product without my permission.