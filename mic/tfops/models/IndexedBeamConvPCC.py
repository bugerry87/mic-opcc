## Installed
import tensorflow as tf
from keras import Model
from keras.layers import (
	Activation,
	Dropout,
	LayerNormalization,
	Concatenate,
	Flatten,
	Dense,
	Add,
	Input,
)

## Local
from ..layers import (
	IndexConv2D as Conv,
	Distiller,
	gather,
)
from .. import bitops

@tf.function(experimental_relax_shapes=True)
def indexing(pos, shifts):
	r = tf.range(tf.shape(pos)[0])
	index = [r]
	for i in range(pos.shape[-1]):
		I = tf.roll(pos, shift=i, axis=-1)
		I //= 1
		I = bitops.left_shift(I, shifts)
		I = tf.reduce_sum(I, axis=-1)	#[5 0 5 2 2 0]	#[a b c d e f]
		I = tf.argsort(I)
		index.append(I)			#scatter#[1 5 3 4 0 2]	#[ac bf ac de de bf]
	index = tf.stack(index)
	inverse = tf.argsort(index, axis=-1)
	return index, inverse

class IndexedBeamConvPCC(Model):
	"""
	"""
	def __init__(self,
		kernels=64,
		convolutions=3,
		head_size=2,
		window_size=3,
		start=0,
		end=12,
		precision=12,
		bins=2,
		dims=3,
		dropout=0.0,
		embedding=None,
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
		if embedding:
			self.embedding = embedding
			self.flatten = Flatten()
			self.concat = Concatenate()
		else:
			self.embedding = None
		self.channeler = Dense(
			convolutions+1,
			activation='softmax',
		)
		self.convs = [
			Conv(
				kernels, (1, window_size),
				padding='same',
				bias_initializer='ones',
				activation='relu',
				name=f'conv_{i}',
				dims=dims+1,
				aggregator=Add(),
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
		target = target[...,None]
		if self.embedding:
			E = self.embedding(pos)
			E = self.flatten(E)
			X = self.concat([E, X]) #* (X[...,-1,None] + 1.5)
		Y = gather(X, target)
		C = self.channeler(Y)
		Y = self.distillers[0](Y) * C[...,0,None]
		index, inverse = indexing(pos, self.shifts)
		for i, (conv, norm, distiller) in enumerate(zip(self.convs, self.norms, self.distillers[1:])):
			X = conv(X[None,None,...], index, inverse)[0,0]
			X = norm(X)
			x = gather(X, target)
			x = self.dropout(x)
			x = distiller(x) * C[...,i+1,None]
			Y += x
		Y = self.activation(Y)
		return Y

__all__ = [
	IndexedBeamConvPCC,
]