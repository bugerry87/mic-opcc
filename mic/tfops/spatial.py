
## Installed
import numpy as np
import tensorflow as tf

## Local
from . import range_like


@tf.function
def xyz2uvd(X):
	x, y, z, etc = X[...,0], X[...,1], X[...,2], X[...,3:]
	pi = tf.where(x > 0.0, np.pi, -np.pi)
	pi = tf.where(y < 0.0, pi, 0.0)
	u = tf.math.atan(x / y) + pi
	d = tf.norm(X, axis=-1)
	v = tf.math.divide_no_nan(z, d)
	v = tf.math.asin(v)
	return tf.concat((u[...,None],v[...,None],d[...,None], etc), axis=-1)

@tf.function
def uvd2xyz(U):
	u, v, d, etc = U[..., 0], U[..., 1], U[..., 2], U[...,3:]
	x = tf.math.sin(u) * d 
	y = tf.math.cos(u) * d 
	z = tf.math.sin(v) * d
	return tf.concat((x[...,None],y[...,None],z[...,None], etc), axis=-1)

def edge_detection_step(X, m, t, M=None, x=None):
	x = range_like(m, dtype=X.dtype) if x is None else x
	'''if (d is None):
		A = tf.roll(X, 1, axis=-2) - X
		B = X - tf.roll(X, -1, axis=-2)
		d = tf.math.reduce_sum((A * B)**2)'''
	s = tf.cumsum(tf.cast(m, tf.int32), axis=-1)-1
	a = tf.where(m)[...,0]
	A = tf.gather(X, a)
	B = tf.roll(A, -1, axis=-2)
	b = tf.roll(a, -1, axis=-1)
	a = tf.cast(tf.gather(a, s), X.dtype)
	b = tf.cast(tf.gather(b, s), X.dtype)
	A = tf.gather(A, s)
	B = tf.gather(B, s)
	w = tf.math.divide_no_nan(x-a, b-a)
	Y = A + (B-A) * w[...,None]
	M = tf.math.reduce_sum((X - Y)**2, axis=-1)
	a = tf.argsort(M)
	b = tf.cast(tf.gather(x, a), a.dtype)
	i = tf.math.segment_max(b, s)
	i = tf.gather(a, i)
	K = tf.gather(M, i)
	m = tf.tensor_scatter_nd_max(m, i[...,None], tf.cast(K>t**2, tf.float32))
	return m, M

#@tf.function
def edge_detection(X, t, M=None, x=None, max_iter=200):
	t = tf.constant(t, dtype=X.dtype)
	x = range_like(X[...,0], dtype=X.dtype) if x is None else x
	m = tf.zeros_like(X[...,0])
	last = tf.cast(tf.reduce_max(x), tf.int32)
	m = tf.tensor_scatter_nd_update(m, [[0], [last]], [1., 1.])
	M = tf.ones_like(X[...,0] * t**2)

	'''if (d is None):
		A = tf.roll(X, 1, axis=-2) - X
		B = X - tf.roll(X, -1, axis=-2)
		d = tf.math.reduce_sum((A * B)**2)'''
	
	def cond(m, M):
		return tf.math.reduce_any(M > t**2)
	
	def body(m, M):
		return edge_detection_step(X, m, t, M, x)

	m, M = tf.while_loop(
		cond, body,
		loop_vars=(m, M),
		maximum_iterations=max_iter
	)
	m = tf.where(m)[...,0]
	return m, M