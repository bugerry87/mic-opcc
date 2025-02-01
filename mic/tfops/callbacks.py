
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


class IndexedConvPCCCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		output=None,
		verbose=False,
		):
		"""
		"""
		super(IndexedConvPCCCallback, self).__init__(**{w:self for w in when})
		self.samples = samples
		self.meta = meta
		self.steps = steps or meta.num_of_files
		self.freq = freq
		self.writer = writer
		self.output = output
		self.verbose = verbose
		self.debug_level = 1
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, epoch = args[:2]
		if epoch % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		count_files = 0
		num_symbols = 0
		metrics = {}
		symbol_list = []
		prob_list = []
		mask_list = []
		self.model.reset_metrics()
		self.filename = None
		
		for sample in self.samples:
			features, _, _, index, mask, layer, symbols, qpos, X, filename = sample
			filename = str(filename.numpy().decode())
			tree_end = tf.reduce_all(layer >= self.model.precision)
			self.filename = filename
			probs = self.model(tf.tuple([features, index, mask]))
			symbol_list.append(symbols)
			prob_list.append(probs * mask)
			mask_list.append(mask)

			if tree_end:
				bit_count = 0
				count_files += 1
				filename = path.join(self.output, re.sub(r'[/\.]', '__', path.splitext(filename)[0]))
				probs = tf.concat(prob_list, axis=0)
				symbols = tf.concat(symbol_list, axis=0)
				mask = tf.concat(mask_list, axis=0)
				symbol_list.clear()
				prob_list.clear()
				mask_list.clear()

				if tfc:
					code = tfc_range_encode(probs[..., 1:], symbols - 1).numpy()
					decoded = tfc_range_decode(probs, code, symbols.shape)
					assert(tf.math.reduce_all(symbols == tf.cast(decoded, symbols.dtype)))
					np.array(code).tofile(filename + '.dbx2.bin')
					bit_count = len(code) * 8.0

				mask = mask.numpy().astype(np.uint8).reshape(-1)
				bpp = (bit_count + len(mask)) / X.shape[-2]
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
				symbols.numpy().astype(np.uint8).tofile(filename + '.sym.bin')
				np.packbits(mask).tofile(filename + '.hst.bin')
				qpos.numpy().tofile(filename + '.pts.bin')
				num_symbols += len(symbols)
				self.debug_level = 0
				if self.steps and self.steps == count_files:
					break
		
		metrics['bpp'] = bpp_sum / count_files
		metrics['bpp_min'] = bpp_min
		metrics['bpp_max'] = bpp_max
		metrics['symbols'] = num_symbols / count_files
		if GPUs:
			metrics['mem_peak'] = tf.config.experimental.get_memory_info('GPU:0')['peak']
		
		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			with self.writer.as_default():
				for name, metric in metrics.items():
					tf.summary.scalar(name, metric, epoch)
			self.writer.flush()
		pass


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
		floor=1e-4,
		test_precision=None,
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
		self.floor = floor
		self.test_precision = test_precision
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
		mask_list = [0] * min(self.test_precision or self.model.precision, self.model.slices[-1])
		max_point = 0
		num_symbols = 0
		acc = 0

		if self.range_coder == 'nrc':
			code = np.zeros([0], np.uint8)
			pointer = 0
			pass
		else:
			symbol_list = []
			prob_list = []
			pass
		
		for sample in self.samples:
			qpos, X, filename, layer, tree_end = sample[-5:]
			i = int(sum(layer.numpy() >= self.model.slices[1:]))
			F = sample[i]
			I = sample[i+self.model.num_models]
			M = sample[self.model.num_models*2]
			symbols = sample[-6]

			filename = str(filename.numpy().decode())
			self.filename = filename
			layer = int(layer)
			mask_list[layer] = M
			max_point = max(len(X), max_point)
			
			next(self.delta_time)
			probs = (self.model.models[i](tf.tuple([F, I, M])) + self.floor) * M
			inference += next(self.delta_time)
			acc += np.sum(np.argmax(probs.numpy(), axis=-1) == symbols.numpy())
			num_symbols += len(symbols)

			if self.range_coder == 'nrc':
				symbols = symbols.numpy().astype(np.uint8)
				cdfs = nrc.prob2cdf(probs[..., 1:].numpy())
				p = pointer % 8
				c, pointer = nrc.encode(symbols - 1, cdfs, pointer=p)
				code = nrc.seemless_concat(code, c, p)
			else:
				prob_list.append(probs)
				symbol_list.append(symbols)
				pass

			if tree_end:
				bit_count = 0
				count_files += 1
				filename = path.join(self.output, re.sub(r'[/\.]', '__', path.splitext(filename)[0]))
				mask = tf.concat(mask_list[1::], axis=0)
				mask = np.packbits(mask.numpy().astype(np.uint8))

				if self.range_coder == 'nrc':
					code, pointer = nrc.finalize(code, pointer)
					np.hstack([mask, code]).tofile(filename + '.mic.bin')
					bit_count = len(code) * 8
					code = np.zeros([0], np.uint8)
					pointer = 0
					pass
				elif self.range_coder == 'tfc':
					probs = tf.concat(prob_list, axis=0)
					symbols = tf.concat(symbol_list, axis=0)
					symbol_list.clear()
					prob_list.clear()
					code = tfc_range_encode(probs[..., 1:], symbols - 1).numpy()
					decoded = tfc_range_decode(probs[..., 1:], code, symbols.shape) + 1
					symbols = tf.cast(symbols, tf.uint8)
					decoded = tf.cast(decoded, tf.uint8)
					if tf.reduce_any(symbols != decoded):
						self.tflog.warning('symbols != decoded')
					np.array(code).tofile(filename + '.mic.bin')
					bit_count = len(code) * 8.0
				else:
					probs = tf.concat(prob_list, axis=0)
					symbols = tf.concat(symbol_list, axis=0)
					probs.numpy().tofile(filename + '.prp.bin')
					symbols.numpy().astype(np.uint8).tofile(filename + '.sym.bin')
					symbol_list.clear()
					prob_list.clear()
					pass

				bpp = bit_count / len(X)
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
				bpop_sum += bit_count / len(qpos)
				qpos.numpy().tofile(filename + '.pts.bin')
				
				self.debug_level = 0
				if self.steps and self.steps == count_files:
					break
		
		metrics['acc'] = acc / num_symbols
		metrics['bpp'] = bpp_sum / count_files
		metrics['bpp/min'] = bpp_min
		metrics['bpp/max'] = bpp_max
		metrics['bpp/output'] = bpop_sum / count_files
		metrics['inference'] = inference / count_files
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
