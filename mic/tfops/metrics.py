
## Installed
import tensorflow as tf
from keras.layers import Reshape
from keras.losses import LossFunctionWrapper, categorical_crossentropy
from keras.metrics import Metric, MeanMetricWrapper, mean_squared_error
from keras.utils.metrics_utils import sparse_categorical_matches


def ragged_cross_entropy(y_true, y_pred, *args):
	return [categorical_crossentropy(
		y_true[i],
		y_pred[i],
		*args
	) for i in range(y_true.shape[0])]

def focal_loss(y_true, y_pred,
	gamma=5.0,
	label_smoothing=0.0,
	**kwargs
	):
	"""
	"""
	y_true *= 1.0 - label_smoothing
	y_true += label_smoothing
	pt = (1.0 - y_true) - y_pred * (1.0 - y_true * 2.0)
	loss = -(1 - pt) ** gamma * tf.math.log(pt)
	loss = tf.math.reduce_mean(loss)
	return loss


def focal_mse(y_true, y_pred, **kwargs):
	return tf.math.log(
		tf.math.reduce_mean(
			mean_squared_error(y_true, y_pred) * tf.norm(y_true, axis=-1)
		)
	)


def combinate(y_true, y_pred, loss_funcs):
	def parse(loss_funcs):
		for loss_func in loss_funcs:
			if 'slices' in loss_func:
				gt = y_true[...,loss_func.slices]
				est = y_pred[...,loss_func.slices]
			else:
				gt = y_true
				est = y_pred
			
			if 'reshape' in loss_func:
				reshape = Reshape(loss_func.reshape)
				gt =  reshape(gt)
				est = reshape(est)
			
			weights = loss_func.weights if 'weights' in loss_func else 1.0
			kwargs = loss_func.kwargs if 'kwargs' in loss_func else dict()

			if 'loss' in loss_func:
				L = loss_func.loss(gt, est, **kwargs) * weights
			else:
				L = loss_func(gt, est, **kwargs) * weights
			yield L
		pass
	
	loss = sum([*parse(loss_funcs)])
	return loss


class RaggedCrossEntropy(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='ragged_cross_entropy',
		**kwargs
		):
		"""
		"""
		super(RaggedCrossEntropy, self).__init__(
			ragged_cross_entropy,
			name=name,
			**kwargs
			)
		pass

class FocalLoss(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='focal_loss',
		**kwargs
		):
		"""
		"""
		super(FocalLoss, self).__init__(
			focal_loss,
			name=name,
			**kwargs
			)
		pass

class FocalMSE(LossFunctionWrapper):
	"""
	"""
	def __init__(self,
		name='focal_mse',
		**kwargs
		):
		"""
		"""
		super(FocalMSE, self).__init__(
			focal_mse,
			name=name,
			**kwargs
			)
		pass


class CombinedLoss(LossFunctionWrapper):
	"""
	"""
	def __init__(self, loss_funcs,
		name='combined_loss',
		**kwargs
		):
		"""
		"""
		super(CombinedLoss, self).__init__(
			combinate,
			name=name,
			loss_funcs = loss_funcs,
			**kwargs
			)
		pass


class RaggedAccuracy(MeanMetricWrapper):
	def __init__(self, name="ragged_accuracy", dtype=None):
		super().__init__(
			lambda y_true, y_pred: [
				sparse_categorical_matches(
					tf.math.argmax(y_true[i], axis=-1), y_pred[i]
				)
				for i in range(3)
			],
			name,
			dtype=dtype,
		)
		pass


class SlicedMetric(Metric):
	"""
	"""
	def __init__(self, metric, slices, **kwargs):
		super(SlicedMetric, self).__init__(name='sliced_' + metric.name, **kwargs)
		self.slices = slices
		self.metric = metric
		pass

	def merge_state(self, metrics):
		return self.metric.merge_state(metrics)

	def update_state(self, y_true, y_pred, sample_weight=None):
		return self.metric.update_state(
			y_true[...,self.slices], 
			y_pred[...,self.slices],  
			sample_weight if sample_weight is not None else None
		)
	
	def result(self):
		return self.metric.result()
	
	def reset_state(self):
		return self.metric.reset_state()
	
	def reset_states(self):
		return self.metric.reset_states()