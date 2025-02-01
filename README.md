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

## Citation

## License

This code is provided by myself for purely non-commercial, research purposes. It may not be used commercially in a product without my permission.