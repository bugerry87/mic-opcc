## Installed
import tensorflow as tf
from keras import Model
from keras.layers import (
	Activation,
	Dropout,
	LayerNormalization,
	Input
)

## Local
from ..layers import (
	BeamConv1D,
	Distiller,
)
from .. import bitops

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
def indexing(pos, offset, shifts):
	r = tf.range(tf.shape(pos)[0])
	index = [r]
	inv = [r]
	for i in range(pos.shape[-1]):
		I = tf.roll(pos, shift=i, axis=-1)
		I //= offset
		I = bitops.left_shift(I, shifts)
		I = tf.reduce_sum(I, axis=-1)	#[5 0 5 2 2 0]	#[a b c d e f]
		a = tf.argsort(I) 				#[1 5 3 4 0 2]	#[b f d e a c]
		I = tf.gather(I, a)				#[0 0 2 2 5 5]
		I = tf.unique(I)[-1]	#[0 2 5]#[0 0 1 1 2 2]	#[bf de ac]
		index.append(I)			#gather #[0 0 1 1 2 2]	#[bf bf de de ac ac]				
		inv.append(a)			#scatter#[1 5 3 4 0 2]	#[ac bf ac de de bf]
	return tf.stack(index), tf.stack(inv)

class IndexedBeamConvPCC(Model):
	"""
	"""
	def __init__(self,
		kernels=64,
		convolutions=3,
		head_size=2,
		window_size=3,
		beam=3,
		start=0,
		end=12,
		precision=12,
		bins=2,
		dims=3,
		dropout=0.0,
		name='IndexedBeamConvPCC',
		**kwargs
		):
		"""
		"""
		super(IndexedBeamConvPCC, self).__init__(name=name or 'IndexedBeamConvPCC', **kwargs)
		self.start = start
		self.end = end
		self.precision = precision
		self.bins = bins
		self.dims = dims
		self.convolutions = convolutions
		self.shifts = tf.range(self.dims, dtype=tf.int64) * self.precision
		self.offset = (1, beam, beam)
		self.convs = [
			BeamConv1D(
				kernels, window_size,
				padding='same',
				bias_initializer='ones',
				activation='relu',
				name=f'conv_{i}',
				merge=beam>1,
				dims=dims+1
			) for i in range(convolutions)
		]
		self.norms = [LayerNormalization() for _ in range(convolutions)]
		self.distillers = [
			Distiller(
				kernels,
				bins,
				head_size,
				activation='softplus',
				name=f'distiller_{k}',
			)
			for k in range(convolutions+1)
		]
		self.dropout = Dropout(dropout)
		self.activation = Activation('softmax')
		pass

	@property
	def symbol_size(self):
		return 1<<self.dims
	
	def build(self, input_shapes=None, placeholders=None, *args):
		if input_shapes is None:
			placeholders = placeholders or [
				Input(type_spec=tf.TensorSpec((None, self.precision, 8+3), dtype=tf.float32)),
				Input(type_spec=tf.TensorSpec((None, self.dims), dtype=tf.int64)),
				Input(type_spec=tf.TensorSpec((None), dtype=tf.int32)),
			]
		else:
			placeholders = [
				Input(type_spec=tf.TensorSpec(input_shapes[0], dtype=tf.float32)),
				Input(type_spec=tf.TensorSpec(input_shapes[1], dtype=tf.int64)),
				Input(type_spec=tf.TensorSpec(input_shapes[2], dtype=tf.int32)),
			]
		self._build_input_shape = [
			p.shape for p in placeholders
		]
		self.built = True
		pass
	
	def call(self, inputs, *args, **kwargs):
		X, pos, target = inputs
		Y = tf.gather(X, target)
		Y = self.distillers[0](Y)
		index = indexing(pos, self.offset, self.shifts)[-1]
		for conv, norm, distiller in zip(self.convs, self.norms, self.distillers[1:]):
			X = conv(X, index, index)
			X = norm(X)
			x = tf.gather(X, target)
			x = self.dropout(x)
			x = distiller(x)
			Y += x
		Y = self.activation(Y)
		return Y

__all__ = [
	IndexedBeamConvPCC,
	featurize,
	labelize,
]