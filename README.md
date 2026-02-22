# MIC-OPCC: Multi-Indexed Convolution model for Octree Point Cloud Compression

Multi-Indexed Convolution introduces an alternative approach to spatial feature extraction, which we use in our entropy model to compress the occupation symbols of octree-encoded point clouds. This method offers reduced time and memory usage per point compared to related work.

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
MultiIndexedTransposedConvolutionPCC

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
  --offset Float [Float ...]
                        Quantization offset
  --scale Float [Float ...]
                        Quantization scale
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
  --shuffle INT         Size of the shuffle buffer
  --precision INT, -P INT
                        Quantization precision
  --tree_type INT, -t INT
                        Tree type: 1 = binary tree, 2 = quatree, 3 = octree
  --qmode STR, -q STR   Quantization precision
  --derotate            Sort axis by major components
  --disolver            Run in disolver mode
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
  --beam INT [INT ...], -b INT [INT ...]
                        size of the beam search
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

### MIC-OPCC v2.0

Baulig, G.; Guo, J.-I. Autoregressive and Residual Index Convolution Model for Point Cloud Geometry Compression. Sensors 2026, 26, 1287. https://doi.org/10.3390/s26041287

```
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