
## Build In
import os.path as path
import pickle
import re

## Installed
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, LambdaCallback

## Local
from . import GPUs
from .. import numba_range_coder as nrc
from ..utils import time_delta

## Optional
try:
	import tensorflow_compression as tfc

	#@tf.function(experimental_relax_shapes=True)
	def tfc_range_encode(probs, symbols):
		symbols = tf.reshape(symbols, [-1])
		symbols = tf.cast(symbols, tf.int16)
		probs = tf.reshape(probs, [-1, 255])

		cdf = tf.math.cumsum(probs, axis=-1)
		cdf = tf.math.round(cdf / cdf[...,-1,None] * float(1<<16))
		cdf = tf.cast(cdf, tf.int32)
		cdf = tf.pad(cdf, [(0,0),(1,0)])

		#cdf = tfc.ops.pmf_to_quantized_cdf(probs, precision=16)
		code = tfc.python.ops.gen_ops.range_encode(
			symbols,
			cdf,
			precision=16,
			debug_level=0,
		)
		return code
	
	#@tf.function(experimental_relax_shapes=True)
	def tfc_range_decode(probs, code, shape):
		probs = tf.reshape(probs, [-1, 255])

		cdf = tf.math.cumsum(probs, axis=-1)
		cdf = tf.math.round(cdf / cdf[...,-1,None] * float(1<<16))
		cdf = tf.cast(cdf, tf.int32)
		cdf = tf.pad(cdf, [(0,0),(1,0)])

		#cdf = tfc.ops.pmf_to_quantized_cdf(probs, precision=16)
		symbols = tfc.python.ops.gen_ops.range_decode(
			code,
			shape,
			cdf,
			precision=16,
			debug_level=0,
		)
		return symbols
	
	@tf.function(experimental_relax_shapes=True)
	def binary_range_encode(probs, symbols, hist, debug_level=0):
		mask = tf.cast(hist != 0.0, tf.int32)
		mask *= tf.cast(hist != 1.0, tf.int32)
		shape = tf.reduce_sum(mask)[...,None]
		mask = tf.where(mask)
		symbols = tf.gather_nd(symbols, mask)
		symbols = tf.cast(symbols, tf.int16)
		probs = tf.gather_nd(probs - 0.5, mask)[...,None]
		probs = tf.abs(probs + (-.5, .5))

		cdf = tfc.ops.pmf_to_quantized_cdf(probs, precision=16)
		code = tfc.python.ops.gen_ops.range_encode(
			symbols,
			cdf,
			precision=16,
			debug_level=debug_level,
		)
		if debug_level:
			decoded = tfc.python.ops.gen_ops.range_decode(
				code,
				shape,
				cdf,
				precision=16,
				debug_level=debug_level,
			)
			decoded = tf.cast(decoded, tf.int16)
			tf.assert_equal(symbols, decoded)
		return code
except:
	tfc = None

try:
	import py7zr

	def xor_encode(probs, symbols, hist, filename):
		arcname = filename + '.7z'
		
		mask = hist.numpy()
		mask = np.where((mask != 0.0) * (mask != 1.0))
		symbols = symbols.numpy()[mask]
		probs = probs.numpy()[mask]
		tail = (8 - len(probs)) % 8
		symbols = np.pad(symbols, [0, tail]).reshape([-1, 8])
		probs = np.pad(probs, [0, tail]).reshape([-1, 8])

		symbols = np.packbits(symbols >= 0.5).astype(np.uint8)
		probs = np.packbits(probs >= 0.5).astype(np.uint8)
		code = symbols ^ probs
		code.tofile(filename)
		with py7zr.SevenZipFile(arcname, 'w') as z:
			z.write(filename, filename)
		file_size = path.getsize(arcname)
		return code, file_size
except:
	py7zr = None

class MultiIndexedConvPCCCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		output=None,
		range_coder='nrc',
		verbose=False,
		):
		"""
		"""
		super(MultiIndexedConvPCCCallback, self).__init__(**{w:self for w in when})
		self.samples = samples
		self.meta = meta
		self.steps = steps or meta.num_of_files
		self.freq = freq
		self.writer = writer
		self.output = output
		self.range_coder = range_coder
		self.verbose = verbose
		self.debug_level = 1
		self.delta_time = time_delta()
		self.tflog = tf.get_logger()
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, epoch = args[:2]
		if epoch % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		bpop_sum = 0
		count_files = 0
		metrics = {}
		self.model.reset_metrics()
		self.filename = None
		inference = 0
		max_point = 0
		num_symbols = 0
		acc_sum = 0

		if self.range_coder == 'nrc':
			code = np.zeros([0], np.uint8)
			pointer = 0
			pass
		else:
			symbol_list = []
			prob_list = []
			pass
		
		for sample in self.samples:
			Q, X, filename, layer, tree_end = sample[-5:]
			layer = int(layer)
			i = self.model.model_num(layer)
			F = sample[i]
			I = sample[i+self.model.num_models]
			T = sample[i+self.model.num_models*2]
			L = sample[i+self.model.num_models*3]
			symbols = tf.where(L)[...,-1]

			filename = str(filename.numpy().decode())
			self.filename = filename
			max_point = max(len(X), max_point)
			
			next(self.delta_time)
			probs = self.model.models[i](tf.tuple([F, I, T]))
			inference += next(self.delta_time)
			acc_sum += np.sum(tf.argmax(probs, axis=-1) == symbols)
			num_symbols += len(L)

			if self.range_coder == 'nrc':
				symbols = tf.reshape(symbols, [-1]).numpy().astype(np.uint8)
				cdfs = nrc.prob2cdf(probs.numpy())
				p = pointer % 8
				c, pointer = nrc.encode(symbols, cdfs, pointer=p)
				code = nrc.seemless_concat(code, c, p)
			else:
				prob_list.append(probs)
				symbol_list.append(symbols)
				pass

			if tree_end:
				bit_count = 0
				count_files += 1
				filename = path.join(self.output, re.sub(r'[/\.]', '__', path.splitext(filename)[0]))

				if self.range_coder == 'nrc':
					code, pointer = nrc.finalize(code, pointer)
					bit_count = len(code) * 8
					code = np.zeros([0], np.uint8)
					pointer = 0
					pass
				elif self.range_coder == 'tfc':
					probs = tf.concat(prob_list, axis=0)
					symbols = tf.concat(symbol_list, axis=0)
					symbol_list.clear()
					prob_list.clear()
					symbols = tf.reshape(symbols, [-1])
					code = tfc_range_encode(probs, symbols).numpy()
					decoded = tfc_range_decode(probs, code, symbols.shape)
					symbols = tf.cast(symbols, tf.uint8)
					decoded = tf.cast(decoded, tf.uint8)
					if tf.reduce_any(symbols != decoded):
						self.tflog.warning('symbols != decoded')
					np.array(code).tofile(filename + '.mic.bin')
					bit_count = len(code) * 8.0
				else:
					probs = tf.concat(prob_list, axis=0)
					symbols = tf.concat(symbol_list, axis=0)
					symbols = tf.reshape(symbols, [-1])
					probs = tf.gather(probs, symbols, batch_dims=1)
					bit_count = np.sum(-np.log2(probs.numpy()))
					probs.numpy().tofile(filename + '.prb.bin')
					symbols.numpy().astype(np.uint8).tofile(filename + '.sym.bin')
					symbol_list.clear()
					prob_list.clear()
					pass

				bpp = bit_count / len(X)
				bpop = bit_count / len(Q)
				bpp_min = min(bpp_min, bpop)
				bpp_max = max(bpp_max, bpop)
				bpp_sum += bpp
				bpop_sum += bpop
				
				self.debug_level = 0
				if self.steps and self.steps == count_files:
					break
		
		metrics['acc'] = acc_sum / num_symbols
		metrics['bpp'] = bpp_sum / count_files
		metrics['bpp/min'] = bpp_min
		metrics['bpp/max'] = bpp_max
		metrics['bpp/output'] = bpop_sum / count_files
		metrics['inference'] = inference / count_files
		metrics['inference/point'] = inference / count_files / max_point
		if GPUs:
			peak = tf.config.experimental.get_memory_info('GPU:0')['peak']
			metrics['mem_peak'] = peak
			metrics['mem_peak/point'] = peak / max_point
		
		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			with self.writer.as_default():
				for name, metric in metrics.items():
					tf.summary.scalar(name, metric, epoch)
			self.writer.flush()
		pass

class LogCallback(Callback):
	"""
	"""
	def __init__(self, logger):
		super(LogCallback, self).__init__()
		self.logger = logger
		self.msg = None
		pass

	def __call__(self, log):
		self.logger.info("Test: " + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()]))

	def on_epoch_end(self, epoch, log):
		self.msg = "Epoch {}: ".format(epoch+1) + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()])
	
	def on_epoch_begin(self, epoch, log):
		if self.msg:
			self.logger.info(self.msg)
	
	def on_train_end(self, log):
		if self.msg:
			self.logger.info(self.msg)
