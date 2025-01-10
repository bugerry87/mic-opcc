import math

## Installed
import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import (
	Input
)

## Local
from .IndexedConvPCC import (
    IndexedConvPCC,
    labelize,
    serialize,
)
from .. import bitops
from ... import lidar
from ... import utils


@tf.function(experimental_relax_shapes=True)
def featurize_uids(uids, precision, dim, layer):
	mask = tf.range(precision, dtype=layer.dtype) < layer
	mask = tf.cast(mask, tf.float32)
	mask = tf.reshape(mask, (-1, precision, 1))
	shifts = tf.range(precision * dim, dtype=uids.dtype)
	uids = tf.reshape(uids, (-1, 1))
	uids = bitops.right_shift(uids, shifts)
	uids = bitops.bitwise_and(uids, 1)
	uids = tf.reshape(uids, (-1, precision, dim))
	shifts = tf.range(dim, dtype=uids.dtype)
	features = bitops.left_shift(uids, shifts)
	features = tf.reduce_sum(features, axis=-1)
	features = tf.one_hot(features, 1<<dim) * mask
	features = tf.reshape(features, (-1, (1<<dim) * precision))
	uids = tf.cast(uids, tf.float32)
	return features, uids

@tf.function(experimental_relax_shapes=True)
def featurize_symbols(symbols, dim):
	symbols = tf.reshape(symbols, [-1, 1])
	symbols = bitops.right_shift(symbols, tf.range(1<<dim, dtype=symbols.dtype))
	symbols = bitops.bitwise_and(symbols, 1)
	symbols = tf.cast(symbols, tf.float32)
	unfold = tf.where(symbols)[...,0]
	#symbols = tf.gather(symbols, unfold)
	return symbols, unfold

@tf.function(experimental_relax_shapes=True)
def indexing(uids):
	index = tf.concat([
		*(
			serialize(tf.roll(uids, shift=i, axis=-1))
			for i in range(uids.shape[-1])
		),
	], axis=-1)
	index = tf.argsort(index, axis=0)
	return index


class MultiIndexedConvPCC(Model):
	def __init__(
		self,
		kernels=256,
		precision=12,
		slices=[0,4,8,12],
		convolutions=[4,8,12],
		heads=[1,2,3],
		strides=[1,6,12],
		dropout=0,
		name='MultiIndexedConvPCC',
		**kwargs
	):
		super(MultiIndexedConvPCC, self).__init__(name=name or 'MultiIndexedConvPCC', **kwargs)
		self.precision = precision
		self.slices = np.array(slices)
		self.models = [
			IndexedConvPCC(
				kernels=kernels,
				convolutions=c,
				heads=h,
				precision=precision,
				start=s,
				end=e,
				strides=st,
				dropout=dropout,
				name=f"SubModel_{i}"
			)
			for i, (s, e, c, h, st) in enumerate(zip(slices[:-1], slices[1:], convolutions, heads, strides))
		]
		pass

	def set_meta(self, index, **kwargs):
		"""
		"""
		meta = utils.Prototype(
			symbol_size=self.symbol_size,
			bins=self.bins,
			dtype=self.dtype,
			**kwargs
			)
		if isinstance(index, str) and index.endswith('.txt'):
			meta.index = index
			with open(index, 'r') as fid:
				meta.num_of_files = sum(len(L) > 0 for L in fid)
		else:
			meta.index = [f for f in utils.ifile(index)]
			meta.num_of_files = len(index)
		self.meta = meta
		return meta
	
	@property
	def num_models(self):
		return len(self.models)

	@property
	def dims(self):
		return 3

	@property
	def symbol_size(self):
		return 1<<self.dims
	
	@property
	def feature_shape(self):
		return [self.precision * self.symbol_size + self.bins + self.bins]
	
	@property
	def bins(self):
		return 1<<self.symbol_size
	
	def select_model(self, layer):
		i = sum(layer >= self.slices[1:])
		return self.models[i]

	def parser(self, index,
		dim=3,
		xtype='float32',
		xshape=(-1,4),
		xformat='raw',
		shuffle=0,
		take=0,
		**kwargs
	):
		meta = self.set_meta(
			index,
			dim=dim,
			xtype=xtype,
			xshape=xshape,
			**kwargs
		)
		
		def parse_raw(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, xshape)[...,:meta.dim]
			return X, filename

		if isinstance(index, str) and index.endswith('.txt'):
			parser = tf.data.TextLineDataset(index)
		else:
			parser = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			if take:
				parser = parser.batch(take)
				parser = parser.shuffle(shuffle)
				parser = parser.unbatch()
			else: 
				parser = parser.shuffle(shuffle)
		if take:
			parser = parser.take(take)
		if xformat == 'raw':
			parser = parser.map(parse_raw)
		return parser, meta
	
	def encoder(self, *args,
		parser=None,
		meta=None,
		augmentation=False,
		qmode=bitops.QMODE_CORNERED,
		precision=None,
		**kwargs
	):
		def encoding():
			dummy_feature = tf.zeros([1, *self.feature_shape], dtype=tf.float32)
			dummy_index = tf.zeros([1, self.dims+1], dtype=tf.int32)
			dummy_label = tf.zeros([1, self.bins], dtype=tf.float32)

			for args in parser:
				if isinstance(args, tuple):
					X, filename = args
				else:
					filename = args
					X = lidar.load(str(filename.numpy().decode()), self.meta.xshape, self.meta.xtype)
					X = X[...,:meta.dim]
					pass

				if augmentation:
					a = tf.random.uniform([], -math.pi, math.pi)
					M = tf.concat([
						[[tf.cos(a), -tf.sin(a), 0.]],
						[[tf.sin(a), tf.cos(a), 0.]],
						[[0., 0., 1.]],
					], axis=0)
					X = tf.matmul(X, M)
					X += tf.random.normal(X.shape, 0.0, 0.01)
				dim = X.shape[-1]
				Q, offset, scale, _ = bitops.quantization(X, self.precision, qmode)
				Q, _ = bitops.serialize(Q, self.precision)
				qpos = bitops.realize(Q, self.precision, dim, offset, scale)
				ancestors = tf.zeros([self.symbol_size, self.bins])

				for layer in tf.range(self.slices[0], precision, dtype=tf.int64):
					shift = (self.precision - layer) * self.dims
					symbols, _, uids, _ = bitops.encode(Q, shift, left_aligned=True)
					step = min(len(symbols), self.select_model(int(layer)).strides)
					weights = tf.cast((layer >= self.slices[:-1]) & (layer < self.slices[1:]), dtype=tf.float32)
					gt_labels, _, mask = labelize(symbols, self.bins)
					unfold = featurize_symbols(symbols, self.dims)[-1]

					for i in range(step):
						gt = symbols[i::step]
						labels = gt_labels[i::step]
						features, index = featurize_uids(
							uids[i::step],
							self.precision,
							self.dims,
							layer,
						)
						siblings = gt_labels[max(i-1, 0)::step][:len(features)] * float(i > 0)
						features = tf.concat([features, ancestors[i::step][:len(features)], siblings], axis=-1)
						index = indexing(index)
						index = tf.concat([
							tf.range(len(index), dtype=index.dtype)[...,None],
							index
						], axis=-1)

						features = [
							features if w else dummy_feature
							for w in weights
						]
						index = [
							index if w else dummy_index
							for w in weights 
						]
						labels = [
							labels if w else dummy_label
							for w in weights 
						]

						tree_end = i+1 == step and layer+1 == precision
						yield (
							*features,
							*index,
							mask,
							weights,
							*labels,
							gt,
							qpos, X, filename, layer, tree_end,
						)
						pass

					ancestors = tf.gather(gt_labels, unfold)
					pass
				pass
			pass
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		meta.label_size = self.bins
		meta.precision = self.precision
		precision = min(self.slices[-1], self.precision if precision is None else precision)
		encoder = tf.data.Dataset.from_generator(
			encoding,
			output_signature=(
				*(
					tf.TensorSpec([None, *self.feature_shape], tf.float32)
					for _ in range(self.num_models)
	  			),
				*(
					tf.TensorSpec([None, 4], tf.int32)
					for _ in range(self.num_models)
				),
				tf.TensorSpec([1, self.bins], tf.float32),
				tf.TensorSpec([self.num_models], tf.float32),
				*(
					tf.TensorSpec([None, self.bins], tf.float32)
					for _ in range(self.num_models)
				),
				tf.TensorSpec([None], tf.int64),
				tf.TensorSpec([None, 3], tf.float32),
				tf.TensorSpec([None, 3], tf.float32),
				tf.TensorSpec([], tf.string),
				tf.TensorSpec([], tf.int64),
				tf.TensorSpec([], tf.bool),
			)
		)
		return encoder, meta

	def trainer(self, *args, 
		encoder=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = meta if meta else self.meta
		trainer = encoder.map(
			lambda *inputs: (
				tuple(inputs[:self.num_models*2 + 1]),
				tuple(inputs[self.num_models*2 + 2: self.num_models*3 + 2]),
				tuple([inputs[self.num_models*2 + 1][i] for i in range(self.num_models)])
			)
		)
		return trainer, meta
	
	def validator(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		return self.trainer(*args, encoder=encoder, **kwargs)
	
	def tester(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = meta if meta else self.meta
		tester = encoder
		return tester, meta

	def build(self, *args):
		placeholders = [
			*(
				Input(shape=(*self.feature_shape,), dtype=tf.float32)
				for _ in range(self.num_models)
			),
			*(
				Input(shape=(4,), dtype=tf.int32)
				for _ in range(self.num_models)
			),
			Input(shape=(self.bins,), batch_size=1, dtype=tf.float32),
			Input(shape=[], batch_size=self.num_models, dtype=tf.float32),
		]
		for model in self.models:
			model.build(placeholders=placeholders[:3])
		self._build_input_shape = [
			p.shape for p in placeholders
		]
		self.call(placeholders)
		self.built = True
		pass
	
	def call(self, inputs, *args, **kwargs):
		Y = tuple(
			model(tf.tuple([
				inputs[i],
				inputs[i+self.num_models],
				inputs[self.num_models*2],
			]), *args, **kwargs)
			for i, model in enumerate(self.models)
		)
		return Y

__all__ = [MultiIndexedConvPCC]