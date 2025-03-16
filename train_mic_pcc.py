#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

## Local
from mic.tfops import GPUs
from mic.tfops.models import MultiIndexedConvPCC
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
		description="MultiIndexedConvolutionPCC",
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
		default=1e-4,
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
		'--floor',
		metavar='Float',
		type=float,
		default=1e-4,
		help="Probability floor for range coder (default=1e-4)"
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
		'--qmode', '-q',
		metavar='STR',
		type=str,
		default='centered',
		choices=['centered', 'cornered'],
		help="Quantization precision"
		)
	
	main_args.add_argument(
		'--slices', '-S',
		metavar='INT',
		nargs='+',
		type=int,
		default=[0,4,8,12],
		help="Tree slices"
		)
	
	main_args.add_argument(
		'--kernels', '-k',
		metavar='INT',
		type=int,
		default=256,
		help='num of kernel units'
		)
	
	main_args.add_argument(
		'--convolutions', '-c',
		metavar='INT',
		nargs='+',
		type=int,
		default=[4,8,12],
		help='number of convolution layers'
		)
	
	main_args.add_argument(
		'--heads', '-n',
		metavar='INT',
		nargs='+',
		type=int,
		default=[1,2,3],
		help='number of transformer heads'
		)
	
	main_args.add_argument(
		'--augmentation',
		action='store_true',
		help="Whether to apply data augmentation or (default) not"
		)
	
	main_args.add_argument(
		'--dropout',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help="Dropout (default=0.0)"
		)
	
	main_args.add_argument(
		'--strides', '-s',
		metavar='INT',
		nargs='+',
		type=int,
		default=[1,1,1],
		help="Strid step of each batch (default=[1,1,1])"
		)
	
	main_args.add_argument(
		'--seed',
		metavar='INT',
		type=int,
		default=0,
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
		'--cpu',
		action='store_true',
		help="Whether to allow cpu or (default) force gpu execution"
		)
	
	main_args.add_argument(
		'--checkpoint',
		metavar='PATH',
		help='Load from checkpoint'
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
	learning_rate=0.001,
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
	floor=1e-4,
	shuffle=0,
	slices=[0,4,8,12],
	precision=12,
	qmode=0,
	kernels=256,
	convolutions=[4,8,12],
	heads=[1,2,3],
	augmentation=False,
	dropout=0.0,
	strides=[1,6,12],
	seed=0,
	log_dir='logs',
	verbose=2,
	cpu=False,
	checkpoint=None,
	log_params={},
	**kwargs
	):
	"""
	"""
	if not cpu:
		assert len(GPUs) > 0
		assert tf.test.is_built_with_cuda()
	
	tf.random.set_seed(seed)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(log_dir, timestamp)
	log_model = os.path.join(log_dir, "ckpts", "dbx_tree_{epoch:04d}-{loss:.3f}.weights.h5")
	log_output = os.path.join(log_dir, timestamp + '.log')
	log_data = os.path.join(log_dir, 'test')
	os.makedirs(os.path.join(log_dir, "ckpts"), exist_ok=True)
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
		model = MultiIndexedConvPCC(
			kernels=kernels,
			convolutions=convolutions,
			heads=heads,
			precision=precision,
			slices=slices,
			strides=strides,
			dropout=dropout,
			**kwargs
		)
	
	qmode = 0 if 'centered' else 1
	meta_args = dict(
		xshape=xshape,
		xtype=xtype,
		xformat=xformat,
		qmode=qmode,
	)
	
	trainer, train_meta = model.trainer(train_index, 
		take=steps_per_epoch,
		shuffle=shuffle,
		augmentation=augmentation,
		**meta_args) if train_index else (None, None)
	validator, val_meta = model.validator(val_index,
		take=validation_steps,
		**meta_args) if val_index else (None, None)
	tester, test_meta = model.tester(test_index,
		take=test_steps,
		precision=test_precision,
		**meta_args) if test_index else (None, None)
	master_meta = train_meta or val_meta or test_meta
	auto_steps = master_meta.num_of_files * sum(
		(e - b) * s
		for b, e, s in zip(slices[:-1], slices[1:], strides)
	)
	steps_per_epoch = steps_per_epoch if steps_per_epoch else auto_steps

	if master_meta is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)
	
	loss = tuple(
		CategoricalCrossentropy()
		for _ in range(len(heads))
	)
	metrics = (
		CategoricalAccuracy('acc'),
		#for i in range(len(heads))
	)
	
	model.compile(
		optimizer=Adam(learning_rate=learning_rate),
		loss=loss,
		metrics=metrics,
		#sample_weight_mode='temporal'
		)
	model.build()
	model.summary(print_fn=tflog.info)
	
	if checkpoint and not checkpoint.endswith('.tf'):
		model.load_weights(checkpoint, by_name=checkpoint.endswith('.h5'), skip_mismatch=checkpoint.endswith('.h5'))
	tflog.info("Samples for Train: {}, Validation: {}, Test: {}".format(steps_per_epoch, validation_steps, test_steps))
	
	monitor = monitor or 'val_loss' if validator else 'loss'
	tensorboard = TensorBoard(log_dir=log_dir)
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
			floor=floor,
			test_precision=test_precision,
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
			verbose=verbose
			)
	elif validator is not None:
		history = model.evaluate(
			validator,
			callbacks=callbacks,
			verbose=verbose,
			return_dict=True
			)
	elif tester is not None:
		history = dict()
		test_callback.model = model
		test_callback(history)
		log_callback(history)
	else:
		raise RuntimeError("Unexpected Error!")
	
	tflog.info('Done!')
	return history


if __name__ == '__main__':
	main_args = init_main_args().parse_args()
	main(log_params=main_args.__dict__, **main_args.__dict__)
	exit()