
## Installed
import tensorflow as tf
from keras.layers import (
	Layer,
	Conv2D,
	LayerNormalization,
	Dense,
	Add,
)

@tf.custom_gradient
def permuted_conv(x, W, b, I, I_inv):
	permuted = tf.gather(x, I, axis=-2)
	y = tf.nn.conv2d(permuted, W, strides=1, padding='SAME')
	y = tf.gather(y + b, I_inv, axis=-2)

	def grad(dy):
		dy = tf.gather(dy, I, axis=-2)
		db = tf.reduce_mean(dy, axis=(0,1,2))
		dW = tf.raw_ops.Conv2DBackpropFilter(
			input=permuted,
			filter_sizes=tf.shape(W),
			out_backprop=dy,
			strides=[1,1,1,1],
			padding='SAME'
		)
		dx_perm = tf.raw_ops.Conv2DBackpropInput(
			input_sizes=tf.shape(permuted),
			filter=W,
			out_backprop=dy,
			strides=[1,1,1,1],
			padding='SAME'
		)
		dx = tf.gather(dx_perm, I_inv, axis=-2)
		return dx, dW, db, None, None
	return y, grad

@tf.custom_gradient
def gather(x, I):
	y = tf.gather_nd(x, I)
	def grad(dy):
		return tf.scatter_nd(I, dy, tf.shape(x)), None
	return y, grad

@tf.custom_gradient
def scatter(x, I):
	y = tf.scatter_nd(I, x, tf.shape(x))
	def grad(dy):
		return tf.gather_nd(dy, I), None
	return y, grad

class Distiller(Layer):
	def __init__(
			self,
			kernels,
			bins,
			length,
			name=None,
			**kwargs
		):
		super().__init__(name=name)
		self.blenders = [
			Dense(
				kernels,
		 		activation='relu',
				name=f'blender_{i}',
		 	) for i in range(length)
		]
		self.norms = [
			LayerNormalization()
			for _ in range(length)
		]
		self.distillers = [
			Dense(
				bins,
				name=f'distiller_{i}',
				**kwargs
		 	) for i in range(length+1)
		]
		pass

	def call(self, X, *args, **kwagrs):
		Y = self.distillers[0](X)
		for blender, norm, distiller in zip(self.blenders, self.norms, self.distillers[1:]):
			X = blender(X)
			X = norm(X)
			Y += distiller(X)
		return Y

class PermutedConv2D(Conv2D):
	def call(self, X, I, Inv):
		Y = permuted_conv(X, self.kernel, self.bias, I, Inv)
		Y = self.activation(Y) if self.activation else Y
		return Y

class IndexConv2D(Layer):
	def __init__(
			self,
			filters,
			kernel_size=3,
			dims=3,
			name=None,
			aggregator=Add(),
			**kwargs
		):
		super().__init__(name=name)
		self.aggregator = aggregator
		self.convs = [PermutedConv2D(
			filters,
			kernel_size,
			**kwargs
		) for _ in range(dims)]
		pass

	def call(self, X, index, inverse, **kwargs):
		Y = []
		for i, conv in zip(range(len(index)), self.convs):
			I = index[i]
			Inv = inverse[i]
			x = conv(X, I, Inv)
			Y.append(x)
		return self.aggregator(Y)