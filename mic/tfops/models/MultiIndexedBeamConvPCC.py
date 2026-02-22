import math

## Installed
import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import (
	Input
)

## Local
from . import (
    IndexedBeamConvPCC,
)
from .. import bitops
from ... import lidar
from ... import utils


@tf.function(experimental_relax_shapes=True)
def featurize_origins(uids, precision, dim, layer):
	mask = tf.range(precision * dim, dtype=layer.dtype) < layer * dim
	mask = tf.cast(mask, tf.float32)
	shifts = tf.range(precision * dim, dtype=uids.dtype)
	uids = tf.reshape(uids, (-1, 1))
	uids = bitops.right_shift(uids, shifts)
	uids = bitops.bitwise_and(uids, 1)
	uids = tf.cast(uids, tf.float32)
	features = uids[..., ::-1] - mask * 0.5
	return features

@tf.function(experimental_relax_shapes=True)
def featurize_symbols(symbols, symbol_size, mask):
	mask = tf.cast(mask, tf.float32)
	symbols = tf.reshape(symbols, (-1, 1))
	symbols = bitops.right_shift(symbols, tf.range(symbol_size, dtype=symbols.dtype))
	symbols = bitops.bitwise_and(symbols, 1)
	symbols = tf.cast(symbols, tf.float32) * mask - mask * 0.5
	return symbols

@tf.function(experimental_relax_shapes=True)
def labelize_symbols(symbols, symbol_size):
	symbols = tf.reshape(symbols, [-1, 1])
	symbols = bitops.right_shift(symbols, tf.range(symbol_size, dtype=symbols.dtype))
	symbols = bitops.bitwise_and(symbols, 1)
	symbols = tf.cast(symbols, tf.float32)
	weights = tf.reduce_mean(symbols)
	weights = tf.abs(symbols - weights)
	return symbols, weights

def tail(args):
	for i in args:
		yield i
	while True:
		yield i

class MultiIndexedBeamConvPCC(Model):
	def __init__(
		self,
		kernels=[32],
		precision=12,
		slices=[0,12],
		convolutions=[12],
		neck_length=[2],
		windows=[3],
		beam=[1],
		dropout=0,
		bins=2,
		dims=3,
		tree_type=3,
		name='MultiIndexedBeamConvPCC',
		**kwargs
	):
		super(MultiIndexedBeamConvPCC, self).__init__(name=name or 'MultiIndexedBeamConvPCC', **kwargs)
		self.bins = bins
		self.dims = dims
		self.tree_type = tree_type
		self.precision = precision
		self.slices = np.array(slices)
		self.models = [
			IndexedBeamConvPCC(
				kernels=k,
				convolutions=c,
				head_size=n,
				beam=b,
				window_size=w,
				precision=precision,
				start=s,
				end=e,
				bins=bins,
				dropout=dropout,
				name=f"SubModel_{i}"
			)
			for i, (s, e, k, c, w, n, b) in enumerate(zip(
				slices[:-1],
				slices[1:],
				tail(kernels),
				tail(convolutions),
				tail(windows),
				tail(neck_length),
				tail(beam)
			))
		]
		pass

	def set_meta(self, index, **kwargs):
		"""
		"""
		meta = utils.Prototype(
			symbol_size=self.symbol_size,
			bins=self.bins,
			dtype=self.dtype,
			**kwargs
			)
		if isinstance(index, str) and index.endswith('.txt'):
			meta.index = index
			with open(index, 'r') as fid:
				meta.num_of_files = sum(len(L) > 0 for L in fid)
		else:
			meta.index = [f for f in utils.ifile(index)]
			meta.num_of_files = len(meta.index)
		self.meta = meta
		return meta
	
	@property
	def num_models(self):
		return len(self.models)

	@property
	def symbol_size(self):
		return 1<<self.tree_type 
	
	@property
	def feature_shape(self):
		return [self.precision * self.dims + self.symbol_size + 1]
	
	def model_num(self, layer):
		return int(sum(int(layer) >= self.slices[1:]))
	
	def select_model(self, layer):
		i = self.model_num(layer)
		return self.models[i]

	def parser(self, index,
		dim=3,
		xtype='float32',
		xshape=(-1,4),
		xformat='raw',
		shuffle=0,
		take=0,
		**kwargs
	):
		meta = self.set_meta(
			index,
			dim=dim,
			xtype=xtype,
			xshape=xshape,
			take=take,
			**kwargs
		)
		
		def parse_raw(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, xshape)[...,:meta.dim]
			return X, filename

		if isinstance(index, str) and index.endswith('.txt'):
			parser = tf.data.TextLineDataset(index)
		else:
			parser = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			if take:
				parser = parser.batch(take)
				parser = parser.shuffle(shuffle)
				parser = parser.unbatch()
			else: 
				parser = parser.shuffle(shuffle)
		if take:
			parser = parser.take(take)
		if xformat == 'raw':
			parser = parser.map(parse_raw)
		return parser, meta
	
	def encoder(self, *args,
		parser=None,
		meta=None,
		salt=0.0,
		pepper=0.0,
		derotate=False,
		disolver=False,
		scale=None,
		offset=None,
		chunk=1,
		qmode=bitops.QMODE_CENTERED,
		precision=None,
		grouping='progressive',
		rotate='',
		**kwargs
	):
		def encoding():
			dummy_feature = tf.zeros([1, *self.feature_shape], dtype=tf.float32)
			dummy_pos = tf.zeros([1, self.dims], dtype=tf.int64)
			dummy_target = tf.zeros([0], dtype=tf.int32)
			dummy_label = tf.ones([1, self.bins], dtype=tf.float32)
			symbol_size = tf.cast(self.symbol_size, tf.int64)
			eye = tf.eye(symbol_size)

			for args in parser:
				if isinstance(args, tuple):
					X, filename = args
				else:
					filename = args
					X = lidar.load(str(filename.numpy().decode()), self.meta.xshape, self.meta.xtype)
					X = X[...,:meta.dim]
					pass

				if derotate:
					e = tf.reduce_max(X, axis=0) - tf.reduce_min(X, axis=0)
					a = tf.argsort(e, direction='DESCENDING')
					X = tf.gather(X, a, axis=1)

				if 'x' in rotate:
					a = tf.random.uniform([], -math.pi, math.pi)
					M = tf.concat([
						[[1., 0., 0.]],
						[[0., tf.cos(a), -tf.sin(a)]],
						[[0., tf.sin(a), tf.cos(a)]],
					], axis=0)
					X = tf.matmul(X, M)
				if 'y' in rotate:
					a = tf.random.uniform([], -math.pi, math.pi)
					M = tf.concat([
						[[tf.cos(a), 0., tf.sin(a)]],
						[[0., 1., 0.]],
						[[-tf.sin(a), 0., tf.cos(a)]],
					], axis=0)
					X = tf.matmul(X, M)
				if 'z' in rotate:
					a = tf.random.uniform([], -math.pi, math.pi)
					M = tf.concat([
						[[tf.cos(a), -tf.sin(a), 0.]],
						[[tf.sin(a), tf.cos(a), 0.]],
						[[0., 0., 1.]],
					], axis=0)
					X = tf.matmul(X, M)
				
				dim = X.shape[-1]
				if qmode is None or qmode == 'none':
					Q = tf.cast(X, tf.int64)
				else:
					Q = bitops.quantization(tf.cast(X, tf.float64), 1<<self.precision, offset=offset, scale=scale, mode=qmode)[0]
				Q = bitops.serialize(Q, self.precision)

				if salt:
					s = (round(len(Q)*salt),)
					r = tf.random.uniform(s, 0, 1<<(self.precision*dim), dtype=tf.int64)
					Q = tf.concat([Q, r], axis=0)
				Q = tf.unique(Q)[0]

				if pepper:
					p = round(len(Q)*pepper)
					Q = tf.random.shuffle(Q)
					Q = Q[p:]
				Q = tf.sort(Q)
				q = len(Q) // chunk

				for step in tf.range(self.slices[0] * self.dims, precision * self.dims, self.tree_type, dtype=tf.int64):
					layer = step // self.tree_type
					shift = self.precision * self.dims - step
					symbols, _, uids, _ = bitops.encode(Q, shift, dim=self.tree_type, left_aligned=True)
					sub = tf.cast((layer >= self.slices[:-1]) & (layer < self.slices[1:]), dtype=tf.float32)

					C = [i for i in range(0, len(symbols), q)]
					for c in C:
						if len(C) > 1:
							S = symbols[c:c+q]
							U = uids[c:c+q]
						else:
							S = symbols
							U = uids
						labels = labelize_symbols(S, symbol_size)[0]
						mask = tf.zeros([len(labels), self.symbol_size])
						
						for g, group in enumerate(groups):
							scatter, skip, *flags = group
							e = tf.gather(eye, flags)
							e = tf.reduce_sum(e, axis=0, keepdims=True)
							e = tf.repeat(e, len(labels), axis=0)
							t = tf.where(e[skip::scatter]) * [[scatter, 1]] + [[skip, 0]]
							m = 1.0-mask if disolver else mask
							target = tf.cast(e + labels + m > 0, tf.int32)
							i = tf.where(target)
							f_origins = tf.gather(U, i[...,0]) + bitops.left_shift(i[...,1], shift - self.tree_type)
							qpos = bitops.realize(f_origins, self.precision, dim, 0, 1, xtype=tf.int64)
							f_origins = featurize_origins(
								f_origins,
								self.precision,
								self.dims,
								layer+1,
							)
							M = tf.gather(mask, i[...,0])
							f_symbols = tf.gather(S, i[...,0])
							f_symbols = featurize_symbols(f_symbols, symbol_size, M)
							features = tf.concat([f_origins, f_symbols, tf.gather_nd(f_symbols, i)[...,None]], axis=-1)
							target = tf.reshape(target, [-1])
							target = tf.cumsum(target) - 1
							target = tf.reshape(target, [-1, self.symbol_size])
							target = tf.gather_nd(target, t)
							L = tf.gather_nd(labels, t)[...,None]
							L = [1.0, 0.0] + L * [-1.0, 1.0]
							
							F = [
								features if s else dummy_feature
								for s in sub
							]
							I = [
								qpos if s else dummy_pos
								for s in sub 
							]
							T = [
								target if s else dummy_target
								for s in sub 
							]
							L = [
								L if s else dummy_label
								for s in sub 
							]

							tree_end = c == C[-1] and step+self.tree_type == precision*self.dims and g+1 == len(groups)
							yield (
								*F, *I, *T, *L, sub,
								Q, X, filename, layer, tree_end,
							)
							e = tf.ones(len(t))
							mask = tf.tensor_scatter_nd_update(mask, t, e)
						pass
					pass
				pass
			pass
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		
		if grouping == 'sequential':
			groups = [ # (scatter, offset, *flags)
				[(1,0,0), (1,0,1)],
				[(1,0,0), (1,0,1), (1,0,2), (1,0,3)],
				[(1,0,0), (1,0,1), (1,0,2), (1,0,3), (1,0,4), (1,0,5), (1,0,6), (1,0,7)],
			][self.tree_type-1]
		elif grouping == 'progressive':
			groups = [ # (scatter, offset, *flags)
				[(2,0,0), (2,1,0), (1,0,1)],
				[(2,0,0), (2,1,0), (1,0,3), (1,0,1,2)],
				#[(2,0,0), (2,1,0), (2,1,7), (2,0,7), (1,0,3), (1,0,4), (1,0,1,6), (1,0,2,5)],
				#[(2,0,0), (2,1,0), (1,0,7), (1,0,2), (1,0,5), (1,0,6), (1,0,1), (1,0,3), (1,0,4)],
				[(1,0,0), (1,0,7), (1,0,1), (1,0,6), (1,0,5), (1,0,2), (1,0,3), (1,0,4)],
			][self.tree_type-1]
		else:
			groups = [
				[(1,0,0,1)],
				[(1,0,0,1,2,3)],
				[(1,0,0,1,2,3,4,5,6,7)],
			][self.tree_type-1]
		precision = min(self.slices[-1], precision or self.precision)
		meta.label_size = self.bins
		meta.precision = self.precision
		meta.grouping = grouping
		steps = sum([z-a for a, z in zip(self.slices[:-1], self.slices[1:])]) + chunk - 1
		meta.samples = meta.take or meta.num_of_files
		meta.steps = meta.samples * steps * len(groups)
		encoder = tf.data.Dataset.from_generator(
			encoding,
			output_signature=(
				*(
					tf.TensorSpec([None, *self.feature_shape], tf.float32)
					for _ in range(self.num_models)
				),
				*(
					tf.TensorSpec([None, self.dims], tf.int64)
					for _ in range(self.num_models)
				),
				*(
					tf.TensorSpec([None], tf.int32)
					for _ in range(self.num_models)
				),
				*(
					tf.TensorSpec([None, self.bins], tf.float32)
					for _ in range(self.num_models)
				),
				tf.TensorSpec([self.num_models], tf.float32),
				tf.TensorSpec([None], tf.int64),
				tf.TensorSpec([None, 3], tf.float32),
				tf.TensorSpec([], tf.string),
				tf.TensorSpec([], tf.int64),
				tf.TensorSpec([], tf.bool),
			)
		)
		return encoder, meta

	def trainer(self, *args, 
		encoder=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = meta if meta else self.meta
		trainer = encoder.map(
			lambda *inputs: (
				tuple(inputs[:self.num_models*3]),
				tuple(inputs[self.num_models*3: self.num_models*4]),
				tuple([inputs[self.num_models*4][i] for i in range(self.num_models)]),
			)
		)
		return trainer, meta
	
	def validator(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		return self.trainer(*args, encoder=encoder, **kwargs)
	
	def tester(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = meta if meta else self.meta
		tester = encoder
		return tester, meta
	
	def generate(self, confidence=0.8):
		eye = tf.eye(self.symbol_size)
		groups = [(2,0,0), (2,1,7), (2,1,0), (2,0,7), (1,0,3), (1,0,4), (1,0,1,6), (1,0,2,5)]
		Q = tf.zeros([1], dtype=tf.int64)
		labels = tf.zeros([1, self.symbol_size])
		for layer in tf.range(self.precision, dtype=tf.int64):
			shift = self.precision * self.dims - layer * self.tree_type - self.tree_type
			model = self.select_model(layer)
			mask = tf.zeros([len(labels), self.symbol_size])
			for scatter, offset, *flags in groups:
				e = tf.gather(eye, flags)
				e = tf.reduce_sum(e, axis=0, keepdims=True) * 2.0
				L = labels * mask + e
				i = tf.where(L)
				t = tf.where(L > 1.5)[offset::scatter]
				f_origins = tf.gather(Q, i[...,0]) + bitops.left_shift(i[...,-1], shift)
				qpos = bitops.realize(f_origins, self.precision, self.dims, 0, 1, xtype=tf.int64)
				target = tf.reshape(L, [-1])
				target = tf.cast(target > 0.0, tf.int32)
				target = tf.cumsum(target) - 1
				tagret = tf.reshape(target, [-1, self.symbol_size])
				target = tf.gather_nd(tagret, t)
				f_origins = featurize_origins(
					f_origins,
					self.precision,
					self.dims,
					layer+1,
				)
				L = tf.gather(labels, i[...,0])
				features = tf.concat([f_origins, L], axis=-1)
				P = model(tf.tuple([features, qpos, target]))
				P = tf.cast(confidence < P[...,1], tf.float32)
				labels = tf.tensor_scatter_nd_update(labels, t, P)
				mask = tf.tensor_scatter_nd_update(mask, t, tf.ones(len(t)))
			i = tf.where(labels)
			if len(i) == 0:
				break
			labels = tf.gather(labels, i[...,0]) * 0.0
			Q = tf.gather(Q, i[...,0]) + bitops.left_shift(i[...,-1], shift)
		qpos = bitops.realize(Q, self.precision, self.dims, 0, 1, xtype=tf.float32)
		print(qpos)
		return qpos

	def build(self, *args):
		placeholders = [
			*(
				Input(shape=(*self.feature_shape,), dtype=tf.float32)
				for _ in range(self.num_models)
			),
			*(
				Input(shape=(self.dims,), dtype=tf.int64)
				for _ in range(self.num_models)
			),
			*(
				Input(shape=(), dtype=tf.int32)
				for _ in range(self.num_models)
			)
		]
		for i, model in enumerate(self.models):
			model.build(placeholders=placeholders[i::self.num_models])
		self._build_input_shape = [
			p.shape for p in placeholders
		]
		self.call(placeholders)
		self.built = True
		pass
	
	def call(self, inputs, *args, **kwargs):
		Y = tuple(
			model(tf.tuple(
				inputs[i::self.num_models]
			), *args, **kwargs)
			for i, model in enumerate(self.models)
		)
		return Y

__all__ = [MultiIndexedBeamConvPCC]