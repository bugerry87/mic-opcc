
## Installed
import tensorflow as tf

right_shift = tf.bitwise.right_shift
left_shift = tf.bitwise.left_shift
bitwise_and = tf.bitwise.bitwise_and
bitwise_or = tf.bitwise.bitwise_or
bitwise_xor = tf.bitwise.bitwise_xor
invert = tf.bitwise.invert

QMODE_CENTERED = 0
QMODE_CORNERED = 1


@tf.function(experimental_relax_shapes=True)
def quantization(X, depth, mode=QMODE_CENTERED, dtype=tf.int64):
	unit = tf.cast(1<<depth, X.dtype) - 1
	offset = tf.reduce_min(X, axis=0)
	scale = tf.reduce_max(X, axis=0) - offset
	dimsort = tf.argsort(scale, direction='DESCENDING')
	scale = tf.reduce_max(scale)
	offset = tf.gather(offset, dimsort)
	X = tf.gather(X, dimsort, axis=-1)
	dimsort = tf.argsort(dimsort)

	if mode == QMODE_CENTERED:
		offset -= (scale - (tf.reduce_max(X, axis=0) - offset)) * 0.5

	scale = unit / scale
	X -= offset
	X *= scale
	X = tf.round(X)
	X = tf.cast(X, dtype)
	return X, offset, scale, dimsort

@tf.function(experimental_relax_shapes=True)
def serialize(Q, depth):
	dim = Q.shape[-1]
	shifts = tf.range(depth, dtype=Q.dtype)
	Q = right_shift(Q[...,None], shifts)
	Q = bitwise_and(Q, tf.constant(1, Q.dtype))
	Q = left_shift(Q, tf.range(dim, dtype=Q.dtype)[None,...,None])
	Q = tf.reduce_sum(Q, axis=-2)
	Q = left_shift(Q, shifts * dim)
	Q = tf.reduce_sum(Q, axis=-1)
	Q = tf.sort(Q)
	Q, inv = tf.unique(Q)
	return Q, inv

@tf.function(experimental_relax_shapes=True)
def realize(X, depth, dim, offset=0.0, scale=1.0, xtype=tf.float32):
	one = tf.constant(1, dtype=X.dtype)
	X = tf.reshape(X, [-1, 1])
	X = right_shift(X, tf.range(depth * dim, dtype=X.dtype))
	X = bitwise_and(X, one)
	X = tf.reshape(X, (-1, depth, dim))
	X = left_shift(X, tf.range(depth, dtype=X.dtype)[...,None])
	X = tf.reduce_sum(X, axis=-2)
	X = tf.cast(X, xtype)
	X /= scale
	X += offset
	return X


def sort(X, bits=64, reverse=False, absolute=False, axis=0):
	with tf.name_scope("sort_bits"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		shifts = tf.range(bits, dtype=X.dtype)
		Y = right_shift(X, shifts)
		Y = bitwise_and(Y, one)
		keepdims = axis!=0
		
		p = tf.math.reduce_sum(Y, axis, keepdims)
		if absolute:
			p2 = tf.math.reduce_sum(one-Y, axis, keepdims)
			p = tf.math.reduce_max((p, p2), axis, keepdims)
		p = tf.argsort(p)
		if reverse:
			p = p[::-1]
		p = tf.cast(p, dtype=X.dtype)
		
		X = right_shift(X, p)
		X = bitwise_and(X, one)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, p


def permute(X, p, bits=63):
	with tf.name_scope("permute_bits"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		X = right_shift(X, tf.range(bits, dtype=X.dtype))
		X = bitwise_and(X, one)
		X = left_shift(X, p)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X


def tokenize(X, dim, depth, axis=0):
	with tf.name_scope("tokenize"):
		X = tf.sort(X, axis=axis)
		shifts = tf.range(depth, dtype=X.dtype) * tf.cast(dim, X.dtype)
		tokens = right_shift(X, shifts[::-1])
		tokens = tf.transpose(tokens)
	return tokens

#@tf.function(experimental_relax_shapes=True)
def encode(X, shift, dim=3, left_aligned=False):
	with tf.name_scope("encode"):
		X = tf.reshape(X, [-1])
		shift = tf.reshape(shift, [-1])
		one = tf.constant(1, X.dtype)
		bits = left_shift(one, dim)
		mask = bits - one
		uids = right_shift(X, shift)
		if left_aligned:
			uids = left_shift(uids, shift)
		uids, inv = tf.unique(uids)
		tokens = bitwise_and(right_shift(X, (shift - dim)), mask)
		hist = tf.one_hot(tokens, tf.cast(bits, tf.int32), dtype=X.dtype)
		hist = tf.math.unsorted_segment_sum(hist, inv, inv[-1] + 1)
		symbols = tf.cast(hist > 0, X.dtype)
		symbols = left_shift(symbols, tf.range(bits, dtype=X.dtype))
		symbols = tf.math.reduce_sum(symbols, axis=-1)
	return symbols, hist, uids, inv

#@tf.function(experimental_relax_shapes=True)
def decode(symbols, pos, dim, buffer=tf.constant([0], dtype=tf.int64)):
	with tf.name_scope("decode"):
		size = tf.reshape(tf.size(buffer), [-1])
		symbols = tf.slice(symbols, pos, size)
		symbols = tf.reshape(symbols, (-1,1))
		symbols = right_shift(symbols, tf.range(1<<dim, dtype=symbols.dtype))
		symbols = bitwise_and(symbols, 1)
		x = tf.where(symbols)
		i = x[...,0]
		x = x[...,1]
		X = left_shift(buffer, dim)
		pos += tf.size(X)
		X = x + tf.gather(X, i)
	return X, pos