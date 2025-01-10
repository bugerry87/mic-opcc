
## Installed
import tensorflow as tf

## Config
#tf.config.optimizer.set_experimental_options({
#	'loop_optimization': False,
#	'disable_model_pruning': True,
#	'disable_meta_optimizer': True,
#})

GPUs = tf.config.list_physical_devices('GPU')
#for gpu in GPUs:
#	tf.config.experimental.set_memory_growth(gpu, True)

@tf.function
def normalize(X):
	n = tf.abs(X)
	n = tf.math.reduce_sum(n, axis=-1, keepdims=True)
	X = tf.math.divide_no_nan(X, n)
	return X

@tf.function
def normalized_relu(X):
	X = tf.maximum(X, 0.0)
	n = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	X = tf.math.divide_no_nan(X, n)
	return X

@tf.function
def underscore_relu(X):
	return tf.nn.relu(X) - 1.0

def neg_ones():
	return tf.initializers.Constant(-1.0)


@tf.function
def range_like(X, dtype=None):
	r = tf.ones_like(X)
	r = tf.where(X)
	return tf.cast(r, dtype)


@tf.function
def batched_identity(shape, dtype='float32'):
	return tf.eye(shape[2], shape[3], batch_shape=shape[1:2], dtype=dtype),


@tf.function
def channel_priority(X, dtype=tf.int32):
	Y = tf.zeros_like(X)
	X = tf.abs(X)
	for n in range(X.shape[-1]):
		mask = tf.cast((X != 0) & (X < tf.math.reduce_max(X, axis=-1, keepdims=True)), X.dtype)
		Y += mask 
		X *= mask
	return tf.cast(Y, dtype)


def count(input, dtype=None):
	c = tf.ones_like(input, dtype=dtype)
	c = tf.math.reduce_sum(c)
	return c


def yield_devices(prefer=None):
	devices = tf.python.client.device_lib.list_local_devices()
	devices = [d for d in devices if d.device_type in prefer] or devices
	i = 0
	while True:
		yield devices[i % len(devices)]
		i += 1


@tf.function(experimental_relax_shapes=True)
def split_batch(*args, batch_size=8):
	N = len(args[0])
	i = tf.range(0, N, batch_size)
	return [
		tf.RaggedTensor.from_row_starts(x, i)
		for x in args
	]