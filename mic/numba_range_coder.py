## Installed
import numpy as np
import numba as nb


def prob2cdf(probs, precision=24, floor=0, dtype=np.int64):
	total = (1<<precision) - 1
	probs = np.array(probs, dtype=np.float64)
	probs = np.clip(probs, floor, 1.0)
	shape = [*probs.shape]
	shape[-1] += 1
	cdf = np.zeros(shape, dtype=np.float64)
	cdf[..., 1:] = probs 
	cdf /= np.linalg.norm(cdf, ord=1, axis=-1, keepdims=True)
	cdf = np.cumsum(cdf, axis=-1)
	cdf *= total
	cdf[:,-1] = total
	return cdf.astype(dtype)

@nb.njit
def _encode_stage(low, high, underflow, half_range, quat_range, mask):
	shift = 0
	code = 0
	while low & half_range or not high & half_range:
		bit = low & half_range > 0
		code <<= 1
		code |= bit
		shift += 1
		while underflow:
			code <<= 1
			code |= bit^1
			shift += 1
			underflow -= 1
		low <<= 1
		high <<= 1
		high |= 1
	low &= mask
	high &= mask
	
	while low & ~high & quat_range:
		underflow += 1
		low = low<<1 ^ half_range
		high = (high ^ half_range) << 1 | half_range | 1
	return code, shift, underflow, low, high

@nb.njit
def encode(symbols, cdfs, precision=62, pointer=0):
	total_range = int(1 << precision)
	curr_range = total_range
	half_range = curr_range >> 1
	quat_range = half_range >> 1
	mask = curr_range - 1
	low = 0
	high = mask
	underflow = 0
	n = len(cdfs)
	stages = np.zeros((n, 2), dtype=np.int64)
	
	for i in range(n):
		s = symbols[i]
		start, end, total = int(cdfs[i, s]), int(cdfs[i, s+1]), int(cdfs[i, -1])

		curr_range //= int(total)
		high = low + int(end) * curr_range - 1
		low = low + int(start) * curr_range

		code, shift, underflow, low, high = _encode_stage(low, high, underflow, half_range, quat_range, mask)
		curr_range = high - low + 1
		stages[i] = (code, shift)
		pass

	p = pointer % 8
	buffer_size = int(np.ceil((stages[:,1:].sum() + 1 + p) / 8))
	buffer = np.zeros(buffer_size, np.uint8)
	for code, shift in stages:
		for s in range(shift-1, -1, -1):
			b = (code >> s) & 1
			buffer[p//8] |= b << (p % 8)
			p += 1
			pass
		pass
	return buffer, p

def seemless_concat(buffer, code, pointer):
	if pointer % 8:
		buffer[-1] += code[0]
		code = code[1:]
	buffer = np.hstack((
		buffer,
		code,
	))
	return buffer

def finalize(buffer, pointer: int):
	p = pointer
	buffer[p//8] |= 1 << (p % 8)
	p += 1
	return buffer, p

@nb.njit
def _read(buffer, window: int, mask: int, p: int):
	window = (window<<1) & mask
	if p//8 < len(buffer):
		window |= (buffer[p//8] >> (p % 8)) & 1
	p += 1
	return window, p

@nb.njit
def read(buffer, window: int, shift: int, mask: int, p=0):
	for _ in range(shift):
		window, p = _read(buffer, window, mask, p)
	return window, p

@nb.njit
def _decode_stage(
	buffer,
	low: int,
	high: int,
	half_range: int,
	quat_range: int,
	mask: int,
	window: int,
	p: int
):
	while low & half_range or not high & half_range:
		window, p = _read(buffer, window, mask, p)
		low <<= 1
		high <<= 1
		high |= 1
	low &= mask
	high &= mask
	
	while low & ~high & quat_range:
		window = (window<<1 & mask>>1) | (window & half_range)
		if p//8 < len(buffer):
			window |= (buffer[p//8] >> (p % 8)) & 1
		p += 1
		low = low<<1 ^ half_range
		high = (high ^ half_range) << 1 | half_range | 1
	return window, p, low, high

@nb.njit
def decode(buffer, cdfs, precision=62, pointer=0):
	curr_range = int(1 << precision)
	half_range = curr_range >> 1
	quat_range = half_range >> 1
	mask = curr_range - 1
	symbols = np.zeros((len(cdfs)), dtype=np.int64)
	window, pointer = read(buffer, 0, precision, mask, pointer)
	low = 0

	for i in range(len(cdfs)):
		symbol = 0
		end = len(cdfs[i])
		total = int(cdfs[i, -1])
		offset = window - low
		value = offset // int(curr_range//total)
		assert 0 <= value < total, (value, total, i)

		while end - symbol > 1:
			mid = (symbol + end) // 2
			if value < cdfs[i, mid]:
				end = mid
			else:
				symbol = mid
		
		symbols[i] = symbol
		start, end = cdfs[i, symbol], cdfs[i, symbol+1]

		curr_range //= total
		high = low + int(end) * curr_range - 1
		low = low + int(start) * curr_range
		window, pointer, low, high = _decode_stage(buffer, low, high, half_range, quat_range, mask, window, pointer)
		curr_range = high - low + 1

	return symbols, pointer

def compile(bins=255, size=1025, dtype=np.int64):
	p = np.random.rand(size, bins)
	symbols = np.argmax(p, axis=-1).astype(dtype)
	cdfs = prob2cdf(p)
	code, pointer = encode(symbols, cdfs)
	code, pointer = finalize(code, pointer)
	y, pointer = decode(code, cdfs)
	return code, y, pointer

if __name__ == '__main__':
	from timeit import timeit
	bins = 255
	size = 152267
	repeats = 100

	p = np.random.rand(size, bins)
	p -= p.min(axis=-1, keepdims=True)
	symbols = np.argmax(p, axis=-1)
	cdfs = prob2cdf(p)
	code, pointer = encode(symbols, cdfs)
	code, pointer = finalize(code, pointer)
	y, pointer = decode(code, cdfs)

	assert np.all(symbols == y), ('Best Case', symbols, y)

	print('\n---- Best Case ----')
	print('Compression:', len(code) / size)
	print('Test', repeats, 'times:')
	time = timeit(lambda: encode(symbols, cdfs), number=repeats)
	print('Encoding, total:', time, 'average:', time / repeats)
	time = timeit(lambda: decode(code, cdfs), number=repeats)
	print('Decoding, total:', time, 'average:', time / repeats)

	p = np.random.rand(size, bins)
	symbols = np.random.randint(bins, size=size)
	cdfs = prob2cdf(p, floor=0.01)
	code, pointer = encode(symbols, cdfs)
	code, pointer = finalize(code, pointer)
	y, pointer = decode(code, cdfs)

	assert np.all(symbols == y), ('Random Case', symbols, y)

	print('\n---- Random Case ----')
	print('Compression:', len(code) / size)
	print('Test', repeats, 'times:')
	time = timeit(lambda: encode(symbols, cdfs), number=repeats)
	print('Encoding, total:', time, 'average:', time / repeats)
	time = timeit(lambda: decode(code, cdfs), number=repeats)
	print('Decoding, total:', time, 'average:', time / repeats)
