
## Installed
import tensorflow as tf
from keras.layers import (
	Layer,
	Conv1D,
	Conv2D,
	LayerNormalization,
	Dense,
)


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

class BeamConv1D(Layer):
	def __init__(
			self,
			filters,
			kernel_size=3,
			dims=3,
			name=None,
			merge=True,
			**kwargs
		):
		super().__init__(name=name)
		self.merge = merge
		self.pre = [Conv1D(
			filters,
			kernel_size,
			**kwargs
		) for _ in range(dims)]
		if self.merge:
			self.post = [Conv1D(
				filters,
				kernel_size,
				**kwargs
			) for _ in range(dims)]
		pass

	def call(self, X, index, inv, **kwargs):
		Y = 0
		if self.merge:
			for i, pre, post in zip(range(len(index)), self.pre, self.post):
				V = inv[i]
				I = index[i]
				x = tf.nn.gather(X, V)
				n = tf.reduce_max(I) + 1
				x = pre(x[None,...], **kwargs)[0]
				x = tf.math.unsorted_segment_sum(x, I, n)
				x = post(x[None,...], **kwargs)[0]
				x = tf.nn.gather(x, I)
				Y += x
		else:
			for i, pre in zip(range(len(inv)), self.pre):
				V = inv[i]
				x = tf.gather(X, V)
				x = pre(x[None,...], **kwargs)[0]
				x = tf.scatter_nd(V[...,None], x, tf.shape(x))
				Y += x
		return Y


class BeamConv2D(Layer):
	def __init__(
			self,
			filters,
			kernel_size=3,
			dims=3,
			name=None,
			merge=True,
			**kwargs
		):
		super().__init__(name=name)
		self.merge = merge
		self.pre = [Conv2D(
			filters,
			kernel_size,
			activation='relu',
			**kwargs
		) for _ in range(dims)]
		if self.merge:
			self.post = [Conv2D(
				filters,
				(kernel_size, 1),
				activation='relu',
				**kwargs
			) for _ in range(dims)]
		pass

	def call(self, X, index, inv, **kwargs):
		Y = 0
		if self.merge:
			for i, pre, post in zip(range(len(index)), self.pre, self.post):
				V = inv[i, ..., None]
				x = tf.gather_nd(X, V)
				I = index[i]
				n = tf.reduce_max(I) + 1
				x = pre(x[None,...], **kwargs)[0]
				x = tf.math.unsorted_segment_sum(x, I, n)
				x = post(x[None,...], **kwargs)[0]
				x = tf.gather(x, I)
				Y += x
		else:
			for i, pre in zip(range(len(inv)), self.pre):
				V = inv[i, ..., None]
				x = tf.gather_nd(X, V)
				x = pre(x[None,...], **kwargs)[0]
				x = tf.tensor_scatter_nd_update(x, V, x)
				Y += x
		return Y