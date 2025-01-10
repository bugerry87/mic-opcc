
## Installed
import tensorflow as tf
from keras.layers import (
	Layer,
	Add,
	MultiHeadAttention,
	LayerNormalization,
	Embedding,
	Conv1D,
	Conv2D,
	Dropout,
	Dense,
)

## Local
from . import normalized_relu, normalize

class SeqEmbedding(Layer):
  def __init__(self, vocab_size, max_length, depth):
    super().__init__()
    self.pos_embedding = Embedding(
	    input_dim=max_length,
		output_dim=depth,
	)
    self.token_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=depth,
	)
    self.add = Add()

  def call(self, seq):
    seq = self.token_embedding(seq)
    x = tf.range(tf.shape(seq)[-2])
    x = x[None, :]
    x = self.pos_embedding(x)
    return self.add([seq,x])

class SelfAttention(Layer):
	def __init__(self, key_dim, num_heads=1, **kwargs):
		super().__init__()
		self.mha = MultiHeadAttention(num_heads, key_dim, **kwargs)
		self.add = Add() 
		self.layernorm = LayerNormalization()

	def call(self, x, **kwargs):
		attn = self.mha(
			x, x,
			**kwargs
		)
		x = self.add([x, attn])
		x = self.layernorm(x)
		return x

class CrossAttention(Layer):
	def __init__(self, key_dim, num_heads=1, **kwargs):
		super().__init__()
		self.mha = MultiHeadAttention(
			num_heads,
			key_dim,
			**kwargs
		)
		self.add = Add()
		self.layernorm = LayerNormalization()

	def call(self, x, y, **kwargs):
		x = self.mha(y, x, **kwargs)
		x = normalize(x)
		x = self.add([y, x])
		return self.layernorm(x)

class IndexConv1D(Layer):
	def __init__(
			self,
			filters,
			kernel_size=3,
			layers=1,
			dropout=0.0,
			name=None,
			**kwargs
		):
		super().__init__(name=name)

		self.convs = [
			Conv1D(
				filters,
				kernel_size,
				activation='relu',
				**kwargs
			)
			for i in range(layers)
		]
		
		self.layernorm = LayerNormalization()
		self.dropout = Dropout(dropout)
		pass
	
	def call(self, X, index, axis=1, **kwargs):
		n = index.shape[-1]
		
		def infer(X, index, conv):
			result = []
			strides = conv.strides[0]
			if strides > 1:
				inv = tf.argsort(index[::strides], axis=0)
			else:
				inv = index
			m = tf.reduce_max(inv) + 1

			for i in range(n):
				I = index[...,i]
				x = tf.gather(X, I, axis=axis)
				#x -= tf.roll(x, shift=-1, axis=axis)
				x = conv(x, **kwargs)[0]
				x = self.dropout(x)
				x = tf.math.unsorted_segment_sum(x, inv[...,i], m)[None,...]
				result.append(x)
			return tf.concat(result, axis=-1), inv
		
		X, I = infer(X, index ,self.convs[0])
		for conv in self.convs[1:]:
			x, I = infer(X, I, conv)
			X += x
		X = self.layernorm(X)
		return X, I

class IndexConv2D(Layer):
	def __init__(
			self,
			filters,
			kernel_size=3,
			layers=1,
			dims=3,
			dropout=0.0,
			name=None,
			**kwargs
		):
		super().__init__(name=name)
		self.dims = dims

		self.convs = [
			Conv2D(
				filters,
				kernel_size,
				activation='relu',
				**kwargs
			)
			for i in range(layers)
		]

		self.layernorm = LayerNormalization()
		self.dropout = Dropout(dropout)
		pass
	
	def call(self, X, index, axis=1, **kwargs):
		n = index.shape[-1]
		
		def infer(X, index, conv):
			result = []
			strides = conv.strides[0]
			if strides > 1:
				inv = tf.argsort(index[::strides], axis=0)
			else:
				inv = index
			m = tf.reduce_max(inv) + 1

			for i in range(n):
				I = index[...,i]
				x = tf.gather(X, I, axis=axis)
				x = conv(x, **kwargs)[0]
				x = self.dropout(x)
				x = tf.math.unsorted_segment_sum(x, inv[...,i], m)[None,...]
				result.append(x)
			return tf.concat(result, axis=-1), inv
		
		X, I = infer(X, index ,self.convs[0])
		for conv in self.convs[1:]:
			x, I = infer(X, I, conv)
			X += x
		X = self.layernorm(X)
		return X, I

class FeedForward(Layer):
	def __init__(self, units, dropout=0.1, name=None):
		super().__init__(name=name)
		self.seq = tf.keras.Sequential([
			Dense(units=units, activation='relu'),
			Dropout(rate=dropout),
		])
		self.add = Add()
		self.layernorm = LayerNormalization()

	def call(self, x, **kwargs):
		x = self.add([x, self.seq(x, **kwargs)])
		return self.layernorm(x)

class ConvSeq2D(Layer):
	def __init__(
		self,
		filters,
		kernel_size=3,
		layers=1,
		name=None,
		**kwargs,
		):
		super().__init__(name=name)
		self.seq = tf.keras.Sequential([
			Conv2D(
				filters,
				kernel_size,
				activation=normalized_relu,
				**kwargs
			)
			for i in range(layers)
		])
		self.add = Add()
		self.dense = Dense(
			filters,
			activation=normalize
		)
		self.layernorm = LayerNormalization()

	def call(self, x, **kwargs):
		x = self.add([
			self.seq(x, **kwargs),
			self.dense(x, **kwargs)
		])
		return self.layernorm(x)