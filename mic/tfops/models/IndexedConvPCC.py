import math

## Installed
import tensorflow as tf
from keras import Model
from keras.layers import (
	Dense,
	Flatten,
	LayerNormalization,
	Dropout,
	Input
)

## Local
from ..layers import (
	IndexConv1D,
)
from .. import bitops
from ... import utils

@tf.function(experimental_relax_shapes=True)
def featurize(symbol_buffer, uids, precision, layer, merge):
	n = tf.reduce_max(merge) + 1
	mask = tf.range(precision, dtype=layer.dtype) < layer
	mask = tf.cast(mask, tf.float32)
	mask = tf.reshape(mask, (-1, precision, 1))
	shifts = tf.range(precision*3, dtype=uids.dtype)
	uids = tf.reshape(uids, (-1, 1))
	uids = bitops.right_shift(uids, shifts)
	uids = bitops.bitwise_and(uids, 1)
	uids = tf.reshape(uids, (-1, precision, 3))
	uids = tf.cast(uids, tf.float32)
	feature = uids[..., ::-1, :]

	symbols = tf.math.unsorted_segment_max(symbol_buffer, merge, n)
	symbols = tf.reshape(symbols, [-1, precision, 1])
	symbols = bitops.right_shift(symbols, tf.range(8, dtype=symbols.dtype))
	symbols = bitops.bitwise_and(symbols, 1)
	symbols = tf.cast(symbols, tf.float32)
	feature = tf.concat([symbols, feature], axis=-1)
	return feature - mask * 0.5, uids

@tf.function(experimental_relax_shapes=True)
def labelize(symbols, bins):
	labels = tf.one_hot(symbols, bins, dtype=tf.float32)
	hist = tf.math.reduce_mean(labels, axis=0, keepdims=True)
	mask = tf.cast(hist > 0, tf.float32)
	return labels, hist, mask

@tf.function(experimental_relax_shapes=True)
def indexing(pos):
	pos -= tf.reduce_min(pos, axis=0, keepdims=True)
	mx = tf.reduce_max(pos, axis=0, keepdims=True)
	mx = tf.math.cumprod(mx, exclusive=True)
	pos *= mx
	index = tf.concat([
		tf.reduce_sum(
			tf.roll(pos, shift=i, axis=-1),
			axis=-1,
			keepdims=True
		)
		for i in range(3)
	], axis=-1)
	return index

@tf.function(experimental_relax_shapes=True)
def serialize(X):
	i = tf.transpose(X, [0, 2, 1])
	i = tf.reshape(i, [tf.shape(X)[0], -1])
	n = tf.shape(i)[-1]
	i = tf.cast(i > 0, tf.int64)
	i = bitops.left_shift(i, tf.range(n, dtype=tf.int64))
	i = tf.math.reduce_sum(i, axis=-1, keepdims=True)
	return i

class IndexedConvPCC(Model):
	"""
	"""
	def __init__(self,
		kernels=64,
		convolutions=3,
		heads=3,
		start=0,
		end=12,
		precision=12,
		dropout=0.0,
		strides=1,
		name='IndexedConvPCC',
		**kwargs
		):
		"""
		"""
		super(IndexedConvPCC, self).__init__(name=name or 'IndexedConvPCC', **kwargs)
		self.start = start
		self.end = end
		self.precision = precision
		self.strides = strides
		self.flatten = Flatten()

		if False and strides > 1:
			self.strided_conv = IndexConv1D(
				kernels, 3,
				padding='same',
				layers=1,
				dims=3,
				dropout=dropout,
				strides=strides,
				dilation_rate=strides,
				name='strided_conv'
			)
		else:
			self.strided_conv = None

		if convolutions:
			self.conv_neighbor = IndexConv1D(
				kernels, 3,
				padding='same',
				layers=convolutions,
				dropout=dropout,
				name='conv_neighbor'
			)
		else:
			self.conv_neighbor = None

		self.dense = [
			Dense(
				kernels * 4,
				dtype=self.dtype,
				name=f'dense_{i}',
				activation='relu',
				**kwargs
			) for i in range(heads)
		]

		if len(self.dense):
			self.layernorm = LayerNormalization()

		self.symbol_head = Dense(
			self.bins,
			dtype=self.dtype,
			name='symbol_head',
			activation='softmax',
			**kwargs
		)
		self.dropout = Dropout(dropout)
		pass
	
	@property
	def dims(self):
		return 3

	@property
	def symbol_size(self):
		return 1<<self.dims
	
	@property
	def bins(self):
		return 1<<self.symbol_size

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
	
	def parser(self, index,
		dim=3,
		xtype='float32',
		xshape=(-1,4),
		shuffle=0,
		take=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index,
			dim=dim,
			xtype=xtype,
			xshape=xshape,
			**kwargs
			)
		
		def parse(filename):
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
		return parser.map(parse), meta
	
	def encoder(self, *args,
		parser=None,
		meta=None,
		augmentation=False,
		psize_balance=0,
		qmode=bitops.QMODE_CENTERED,
		**kwargs
	):
		def encoding():
			for X, filename in parser:
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
				Q, offset, scale = bitops.quantization(X, self.precision, qmode)
				Q, _ = bitops.serialize(Q, self.precision)
				qpos = bitops.realize(Q, self.precision, dim, offset, scale)
				
				layer = tf.constant(self.start, tf.int64)
				symbol_buffer = tf.zeros([len(Q), self.precision], tf.int64)
				r = tf.range(len(Q), dtype=layer.dtype)[..., None] * [(1, 0)]
				weight = (len(Q) / psize_balance) if psize_balance else 1.0

				while tf.reduce_any(layer < self.end):
					shift = (self.precision - layer) * dim
					symbols, _, uids, inv = bitops.encode(Q, shift, left_aligned=True)
					features, uids = featurize(symbol_buffer, uids, self.precision, layer, inv)
					labels, hist = labelize(symbols, self.bins)
					mask = tf.cast(hist > 0, tf.float32)
					S = tf.gather(symbols, inv)
					symbol_buffer = tf.tensor_scatter_nd_max(symbol_buffer, r, S)
					index = tf.concat([
						*(
							serialize(tf.roll(uids, shift=i, axis=-1))
							for i in range(uids.shape[-1])
						),
					], axis=-1)
					index = tf.argsort(index, axis=0)
					layer += 1
					r += [(0, 1)]

					yield features, labels, weight, index, mask, symbols, qpos, layer, X, filename
				pass
			pass
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		meta.label_size = self.bins
		meta.start = self.start
		meta.end = self.end
		meta.precision = self.precision
		encoder = tf.data.Dataset.from_generator(encoding,
			output_types=(
				tf.float32,
				tf.float32,
				tf.float32,
				tf.int32,
				tf.float32,
				tf.int64,
				tf.float32,
				tf.int64,
				tf.float32,
				tf.string,
			),
			output_shapes=(
				tf.TensorShape([None, self.precision, 8+3]),
				tf.TensorShape([None, self.bins]),
				tf.TensorShape([]),
				tf.TensorShape([None, 3]),
				tf.TensorShape([None, self.bins]),
				tf.TensorShape([None]),
				tf.TensorShape([None, 3]),
				tf.TensorShape([]),
				tf.TensorShape([None, 3]),
				tf.TensorShape([]),
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
			lambda features, labels, weight, index, mask, *args: ((features, index, mask), labels, weight)
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
	
	def build(self, input_shapes=None, placeholders=None, *args):
		if input_shapes is None:
			placeholders = placeholders or [
				Input(type_spec=tf.TensorSpec((None, self.precision, 8+3), dtype=tf.float32)),
				Input(type_spec=tf.TensorSpec((None, 3), dtype=tf.int32)),
				Input(type_spec=tf.TensorSpec((1, self.bins), dtype=tf.float32)),
			]
		else:
			placeholders = [
				Input(type_spec=tf.TensorSpec(input_shapes[0], dtype=tf.float32)),
				Input(type_spec=tf.TensorSpec(input_shapes[1], dtype=tf.int32)),
				Input(type_spec=tf.TensorSpec(input_shapes[2], dtype=tf.float32)),
			]
		self._build_input_shape = [
			p.shape for p in placeholders
		]
		#self.call(placeholders)
		self.built = True
	
	def call(self, inputs, training=False, *args, **kwargs):
		X, I, M = inputs[:3]
		if self.strided_conv is not None:
			X, I = self.strided_conv(X[None, ...,], I)
			X = X[0]
		if self.conv_neighbor is not None:
			X, I = self.conv_neighbor(X[None, ...], I)
			X = X[0]
		X = self.flatten(X)
		Y = X

		for dense in self.dense:
			X = self.dropout(X)
			X = dense(X)
			Y += X
			pass

		if len(self.dense):
			X = self.layernorm(Y)
			pass
		
		if training:
			M = tf.clip_by_value(M, 0.1, 1.0)
		X = self.symbol_head(X) * M
		return X

__all__ = [
	IndexedConvPCC,
	featurize,
	labelize,
	serialize,
]