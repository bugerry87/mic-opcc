#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay 

## Local
from mic.tfops import bitops
from mic.tfops import GPUs
from mic.tfops.models import MultiIndexedBeamConvPCC
from mic.tfops.callbacks import MultiIndexedConvPCCCallback, LogCallback


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="MultiIndexedTransposedConvolutionPCC",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--train_index', '-X',
		metavar='PATH',
		nargs='*',
		default=None,
		help='A index file to training data'
		)
	
	main_args.add_argument(
		'--val_index', '-Y',
		metavar='PATH',
		nargs='*',
		default=None,
		help='A index file to validation data'
		)
	
	main_args.add_argument(
		'--test_index', '-T',
		metavar='PATH',
		nargs='*',
		default=None,
		help='A index file to test data'
		)
	
	main_args.add_argument(
		'--xshape',
		metavar='SHAPE',
		type=int,
		nargs='+',
		default=(-1, 4),
		help='Shape of the input data'
		)
	
	main_args.add_argument(
		'--xtype',
		metavar='TYPE',
		default='float32',
		help='Type of the input data'
		)
	
	main_args.add_argument(
		'--xformat',
		metavar='FORMAT',
		default='raw',
		help='Format of the input data'
		)
	
	main_args.add_argument(
		'--offset',
		metavar='Float',
		nargs='+',
		type=float,
		default=None,
		help="Quantization offset"
		)
	
	main_args.add_argument(
		'--scale',
		metavar='Float',
		nargs='+',
		type=float,
		default=None,
		help="Quantization scale"
		)
	
	main_args.add_argument(
		'--epochs', '-e',
		metavar='INT',
		type=int,
		default=1,
		help='Num of epochs'
		)
	
	main_args.add_argument(
		'--learning_rate',
		metavar='Float',
		type=float,
		default=1e-3,
		help="Learning rate for the Adam optimizer (default=1e-4)"
		)
	
	main_args.add_argument(
		'--monitor',
		metavar='STR',
		default=None,
		help='Choose the metric to be monitored for checkpoints and early stopping (default=automatic)'
		)

	main_args.add_argument(
		'--save_best_only',
		action='store_true',
		help="Whether to save only best model or (default) not"
		)
	
	main_args.add_argument(
		'--stop_patience',
		metavar='INT',
		type=int,
		default=-1,
		help='The early stopping patience (deactivate = -1)'
		)
	
	main_args.add_argument(
		'--steps_per_epoch',
		metavar='INT',
		type=int,
		default=0,
		help='Define to train on a subset'
		)
	
	main_args.add_argument(
		'--validation_freq',
		metavar='INT',
		type=int,
		default=1,
		help="Validation frequency"
		)
	
	main_args.add_argument(
		'--validation_steps',
		metavar='INT',
		type=int,
		default=0,
		help='Define to validate on a subset'
		)
	
	main_args.add_argument(
		'--test_freq',
		metavar='INT',
		type=int,
		default=1,
		help="Test frequency (default=1)"
		)
	
	main_args.add_argument(
		'--test_steps',
		metavar='INT',
		type=int,
		default=0,
		help='Define for test on a subset'
		)
	
	main_args.add_argument(
		'--test_precision',
		metavar='INT',
		type=int,
		default=None,
		help='Define precision during test'
		)
	
	main_args.add_argument(
		'--range_coder',
		metavar='STR',
		type=str,
		default='nrc',
		choices=['nrc', 'tfc', 'none'],
		help='Select range coder implementation'
		)
	
	main_args.add_argument(
		'--shuffle',
		metavar='INT',
		type=int,
		default=0,
		help="Size of the shuffle buffer"
		)
	
	main_args.add_argument(
		'--precision', '-P',
		metavar='INT',
		type=int,
		default=12,
		help="Quantization precision"
		)
	
	main_args.add_argument(
		'--tree_type', '-t',
		metavar='INT',
		type=int,
		default=3,
		help="Tree type: 1 = binary tree, 2 = quatree, 3 = octree"
		)
	
	main_args.add_argument(
		'--qmode', '-q',
		metavar='STR',
		type=str,
		default='centered',
		choices=['centered', 'cornered', 'none'],
		help="Quantization precision"
		)
	
	main_args.add_argument(
		'--derotate',
		action='store_true',
		help='Sort axis by major components'
		)
	
	main_args.add_argument(
		'--disolver',
		action='store_true',
		help='Run in disolver mode'
		)
	
	main_args.add_argument(
		'--rotate',
		metavar='STR',
		type=str,
		default='',
		help='Random rotation augmentation - use "xyz" (default="")'
		)
	
	main_args.add_argument(
		'--grouping', '-g',
		metavar='STR',
		type=str,
		default='progressive',
		choices=['none', 'sequential', 'progressive'],
		help="Grouping strategy"
		)
	
	main_args.add_argument(
		'--slices', '-S',
		metavar='INT',
		nargs='+',
		type=int,
		default=[0,12],
		help="Tree slices"
		)
	
	main_args.add_argument(
		'--chunk', '-C',
		metavar='INT',
		type=int,
		default=1,
		help="Chunk level"
		)

	main_args.add_argument(
		'--kernels', '-k',
		metavar='INT',
		nargs='+',
		type=int,
		default=[32],
		help='num of kernel units'
		)
	
	main_args.add_argument(
		'--windows', '-w',
		metavar='INT',
		nargs='+',
		type=int,
		default=[3],
		help='window size'
		)
	
	main_args.add_argument(
		'--beam', '-b',
		metavar='INT',
		nargs='+',
		type=int,
		default=[1],
		help='size of the beam search'
		)
	
	main_args.add_argument(
		'--convolutions', '-c',
		metavar='INT',
		nargs='+',
		type=int,
		default=[12],
		help='number of convolution layers'
		)
	
	main_args.add_argument(
		'--head_size', '-n',
		metavar='INT',
		nargs='+',
		type=int,
		default=[2],
		help='the dense layer size after convolution'
		)
	
	main_args.add_argument(
		'--salt',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help="Ratio to add salt to data - adds random points (default=0.0)"
		)
	
	main_args.add_argument(
		'--pepper',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help="Ratio to add pepper to data - removes random points (default=0.0)"
		)
	
	main_args.add_argument(
		'--dropout',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help="Dropout (default=0.0)"
		)
	
	main_args.add_argument(
		'--seed',
		metavar='INT',
		type=int,
		default=1,
		help='Initial model seed'
		)
	
	main_args.add_argument(
		'--log_dir',
		metavar='PATH',
		default='logs',
		help="Model type (default=logs)"
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		metavar='INT',
		type=int,
		default=1,
		help="verbose level (see tensorflow)"
		)
	
	main_args.add_argument(
		'--profiler',
		metavar='INT',
		type=int,
		default=0,
		help="Activate profiler per batch (default=0)"
		)
	
	main_args.add_argument(
		'--cpu',
		action='store_true',
		help="Whether to allow cpu or (default) force gpu execution"
		)
	
	main_args.add_argument(
		'--checkpoint',
		metavar='PATH',
		help='Load from checkpoint'
		)
	
	main_args.add_argument(
		'--generate',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help="Generate a confidence point cloud at the end (default=0.0)"
		)
	return main_args


def main(
	train_index=None,
	val_index=None,
	test_index=None,
	xshape=(-1,4),
	xtype='float32',
	xformat='raw',
	epochs=1,
	learning_rate=1e-3,
	monitor=None,
	save_best_only=False,
	stop_patience=-1,
	steps_per_epoch=0,
	validation_freq=1,
	validation_steps=0,
	test_freq=1,
	test_steps=0,
	test_precision=None,
	range_coder='nrc',
	shuffle=0,
	slices=[0,12],
	precision=12,
	offset=None,
	scale=None,
	tree_type=3,
	chunk=1,
	qmode=0,
	derotate=False,
	disolver=False,
	rotate='',
	grouping='progressive',
	kernels=[32],
	windows=[3],
	beam=[1],
	convolutions=[12],
	head_size=[1],
	salt=0.0,
	pepper=0.0,
	dropout=0.0,
	seed=1,
	log_dir='logs',
	verbose=2,
	cpu=False,
	checkpoint=None,
	generate=0.0,
	profiler=0,
	log_params={},
	**kwargs
	):
	"""
	"""
	if not cpu:
		assert len(GPUs) > 0
		assert tf.test.is_built_with_cuda()
	
	os.environ['PYTHONHASHSEED']=str(seed)
	os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(log_dir, timestamp)
	log_model = os.path.join(log_dir, "ckpts", "mic_{epoch:04d}-{loss:.3f}.weights.h5")
	log_output = os.path.join(log_dir, timestamp + '.log')
	log_data = os.path.join(log_dir, 'test')
	os.makedirs(os.path.join(log_dir, 'ckpts'), exist_ok=True)
	train_index = train_index[0] if train_index and len(train_index) == 1 else train_index
	val_index = val_index[0] if val_index and len(val_index) == 1 else val_index
	test_index = test_index[0] if test_index and len(test_index) == 1 else test_index

	tflog = tf.get_logger()
	tflog.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_output)
	tflog.addHandler(fh)
	tflog.info("Main Args:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in log_params.items()]))
	if kwargs:
		tflog.warning("Unrecognized Kwargs:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in kwargs.items()]))
	
	if checkpoint and checkpoint.endswith('tf'):
		model = tf.keras.models.load_model(checkpoint)
	else:
		model = MultiIndexedBeamConvPCC(
			kernels=kernels,
			convolutions=convolutions,
			neck_length=head_size,
			windows=windows,
			precision=precision,
			slices=slices,
			beam=beam,
			tree_type=tree_type,
			dropout=dropout,
			**kwargs
		)
	
	qmode = bitops.QMODE[qmode]
	meta_args = dict(
		xshape=xshape,
		xtype=xtype,
		xformat=xformat,
		qmode=qmode,
		offset=offset,
		scale=scale,
		grouping=grouping,
		derotate=derotate,
		disolver=disolver,
	)
	
	trainer, train_meta = model.trainer(train_index, 
		take=steps_per_epoch,
		shuffle=shuffle,
		rotate=rotate,
		chunk=chunk,
		salt=salt,
		pepper=pepper,
		**meta_args) if train_index else (None, None)
	validator, val_meta = model.validator(val_index,
		take=validation_steps,
		**meta_args) if val_index else (None, None)
	tester, test_meta = model.tester(test_index,
		take=test_steps,
		precision=test_precision,
		**meta_args) if test_index else (None, None)
	steps_per_epoch = train_meta.steps if train_meta else 0
	validation_steps = val_meta.samples if val_meta else 0
	test_steps = test_meta.samples if test_meta else 0

	if (train_meta or val_meta or test_meta) is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)
	
	loss = tuple(
		CategoricalCrossentropy()
		for _ in range(len(slices)-1)
	)
	metrics = (
		CategoricalAccuracy('acc'),
	)
	
	lrs = ExponentialDecay(
		learning_rate,
		steps_per_epoch or validation_steps or test_steps,
		0.9,
	)
	optimizer = Adam(
		learning_rate=lrs,
		)
	model.compile(
		optimizer=optimizer,
		loss=loss,
		weighted_metrics=metrics,
		)
	model.build()
	model.summary(print_fn=tflog.info)
	
	if checkpoint and not checkpoint.endswith('.tf'):
		model.load_weights(checkpoint, by_name=checkpoint.endswith('.h5'), skip_mismatch=checkpoint.endswith('.h5'))
	tflog.info("Samples for Train: {}, Validation: {}, Test: {}".format(steps_per_epoch, validation_steps, test_steps))
	
	monitor = monitor or 'val_loss' if validator else 'loss'
	tensorboard = TensorBoard(log_dir=log_dir, profile_batch=profiler)
	callbacks = [
		tensorboard,
		ModelCheckpoint(
			log_model,
			save_best_only=save_best_only,
			save_weights_only=(not log_model.endswith('.tf')),
			monitor=monitor,
			),
		TerminateOnNaN()
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor=monitor,
			patience=stop_patience
		))
	
	if tester is not None:
		writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
		when = ['on_test_end' if trainer is None else 'on_epoch_end']
		test_callback = MultiIndexedConvPCCCallback(
			samples=tester,
			meta=test_meta,
			freq=test_freq,
			steps=test_steps,
			when=when,
			writer=writer,
			output=log_data,
			verbose=verbose,
			range_coder=range_coder,
		)
		callbacks.append(test_callback)
	
	log_callback = LogCallback(tflog)
	callbacks.append(log_callback)

	if trainer is not None:
		history = model.fit(
			trainer.repeat(),
			epochs=epochs,
			steps_per_epoch = steps_per_epoch,
			callbacks=callbacks,
			validation_freq=validation_freq,
			validation_data=validator,
			validation_steps=validation_steps,
			verbose=verbose
			)
	elif validator is not None:
		history = model.evaluate(
			validator,
			callbacks=callbacks,
			verbose=verbose,
			return_dict=True,
			)
	elif tester is not None:
		history = dict()
		test_callback.model = model
		test_callback(history)
		log_callback(history)
	else:
		raise RuntimeError("Unexpected Error!")
	
	if generate:
		qpos = model.generate(generate)
		qpos.numpy().tofile(os.path.join(log_dir, 'generated.pts.bin'))
	
	tflog.info('Done!')
	return history


if __name__ == '__main__':
	main_args = init_main_args().parse_args()
	main(log_params=main_args.__dict__, **main_args.__dict__)
	exit()