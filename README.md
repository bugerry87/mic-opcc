# MIC-OPCC: Multi-Indexed Convolution model for Octree Point Cloud Compression

Multi-Indexed Convolution introduces an alternative approach to spatial feature extraction, which we use in our entropy model to compress the occupation symbols of octree-encoded point clouds. This method achieves offers reduced time and memory usage per point compared to related work.

## Requirements

```
python:                 3.10
tensorflow:             2.9.0
tensorflow-probability: 0.15.0
tensorflow-compression: 2.9.2
keras:                  2.9.0
numba:                  0.60
scipy:                  1.12.0
```

## Installation

```
conda -n mic-opcc python=3.10 tensorflow=2.9 keras=2.9 numba=0.60 scipy=1.12
pip tensorflow-probability=0.15.0 tensorflow-compression=2.9.2
```

## Datasets

Download and extract the [Semantic KITTI](https://semantic-kitti.org/) dataset to `./data/semantic-kitti/`.

## Usage

```
MultiIndexedConvolutionPCC

options:
  -h, --help            show this help message and exit
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
  --epochs INT, -e INT  Num of epochs
  --learning_rate Float
                        Learning rate for the Adam optimizer (default=1e-4)
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
  --floor Float         Probability floor for range coder (default=1e-4)
  --shuffle INT         Size of the shuffle buffer
  --precision INT, -P INT
                        Quantization precision
  --qmode STR, -q STR   Quantization precision
  --slices INT [INT ...], -S INT [INT ...]
                        Tree slices
  --kernels INT, -k INT
                        num of kernel units
  --convolutions INT [INT ...], -c INT [INT ...]
                        number of convolution layers
  --heads INT [INT ...], -n INT [INT ...]
                        number of transformer heads
  --augmentation        Whether to apply data augmentation or (default) not
  --dropout FLOAT       Dropout (default=0.0)
  --strides INT [INT ...], -s INT [INT ...]
                        Strid step of each batch (default=[1,6,12])
  --seed INT            Initial model seed
  --log_dir PATH        Model type (default=logs)
  --verbose INT, -v INT
                        verbose level (see tensorflow)
  --cpu                 Whether to allow cpu or (default) force gpu execution
  --checkpoint PATH     Load from checkpoint
```

### Run a Toy Example

Run a Toy Example on `mini_index.txt` that contains one single point cloud sample and test the compression result on the very same sample.
This command starts a training session of 10 epochs with 120 iterations each at a quantization precision of 12bits per dimension.
Each iteration process one octree layer. Hance, 12 iterations are needed to process one sample.
The model is configured to 3 sub-modules. Each sub-module is dedicated to 4 octree layers.

- Sub-module 1 applies 4 convolutions and 1 fully-connected layer.
- Sub-module 2 applies 8 convolutions and 2 fully-connected layers.
- Sub-module 3 applies 12 convolutions and 3 fully-connected layers.

This session allows to be run on CPU and uses an Arithmetic Range Coder implementation based on Numba.

```
python ./train_mic_pcc.py -X ./samples/mini_index.txt -T ./samples/mini_index.txt -P 12 -e 10 --steps_per_epoch 120 -S 0 4 8 12 -c 4 8 12 -n 1 2 3 --range_coder=nrc --cpu
```

## Citation

## License

This code is provided by myself for purely non-commercial, research purposes. It may not be used commercially in a product without my permission.