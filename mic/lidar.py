"""
Operations for LiDAR data.

Author: Gerald Baulig
"""

## Build in
import os.path as path

## Installed
import numpy as np

## Local
from . import spatial
from . import bitops

## Optional
try:
	import pcl
except:
	pcl = None
	pass


def psnr(A, B=None, peak=1.0):
	if B is None:
		MSE = A
	else:
		MSE = np.mean((A - B) ** 2)
	
	return 10 * np.log10(float(peak) / MSE)


def snr(A, B=None, noise=1.0):
	if B is None:
		MSE = A
	else:
		MSE = np.mean((A - B) ** 2)
	
	return 10 * np.log10(MSE / float(noise))


def xyz2uvd(X, norm=False, z_off=0.0, d_off=0.0, mode='sphere'):
	x, y, z = X.T
	uvd = np.empty(X.shape)
	with np.errstate(divide='ignore', over='ignore'):
		uvd[:,2] = np.linalg.norm(X, axis=-1)
		uvd[:,0] = np.arctan2(x, y)

		if mode == 'sphere':
			uvd[:,1] = np.arcsin((z + z_off) / (uvd[:,2] + d_off))
		elif mode == 'cone':
			uvd[:,1] = (z + z_off) / (np.linalg.norm(X[:,:2], axis=-1) * d_off)
		else:
			raise ValueError("Unknown mode: '{}'!".format(mode))
	if norm is False:
		pass
	elif norm is True:
		uvd = spatial.prob(uvd)
	else:
		uvd[:,norm] = spatial.prob(uvd[:,norm])
	return uvd


def uvd2xyz(U, z_off=0.0, d_off=0.0, mode='sphere'):
	u, v, d = U.T
	xyz = np.empty(U.shape)
	c = np.cos(v) * (d + z_off) 
	xyz[:,0] = np.sin(u) * c
	xyz[:,1] = np.cos(u) * c
	if mode == 'sphere':
		xyz[:,2] = np.sin(v) * (d + d_off) - z_off
	elif mode == 'cone':
		xyz[:,2] = v * (d + d_off) + z_off # * (np.linalg.norm(xyz[:,:2], axis=-1) + r_off) + z_off
	else:
		raise ValueError("Unknown mode: '{}'!".format(mode))
	return xyz


def dot_keypoints(X, m=1, o=0):
	assert(m > 0)
	assert(o >= 0)
	mask = np.zeros(len(X), dtype=bool)
	P = iter(X)
	k = next(P)
	p0 = next(P)
	p0k = p0 - k
	p0km = spatial.magnitude(p0k)
	mag = p0km
	m = m**2
	mask[0] = True
	mask[-1] = True
	
	for i, p1 in enumerate(P):
		pp = p1 - p0
		ppm = spatial.magnitude(pp)
		mag += ppm
		
		p1k = p1 - k
		p1km = spatial.magnitude(p1k)
		dot = np.dot(p0k, p1k)**2 / (p0km * p1km)
		
		if dot < 1 - np.exp(-o-mag/m)**4:
			#new keypoint detected
			k = p0
			p0 = p1
			p0k = pp
			p0km = ppm
			mag = ppm
			mask[i+1] = True
		else:
			#update
			p0 = p1
			p0k = p1k
			p0km = p1km
	return X[mask], mask


def save(X, output, format=None):
	if format is None:
		output, format = path.splitext(output)
		format = format.split('.')[-1]
	if not format:
		format = 'bin'

	output = "{}.{}".format(output, format)
	if 'bin' in format:
		X.tofile(output)
	elif 'npy' in format:
		np.save(output, X)
	elif pcl and 'ply' in format or 'pcd' in format:
		if X.shape[-1] == 3:
			P = pcl.PointCloud(X)
		elif X.shape[-1] == 4:
			P = pcl.PointCloud_PointXYZI(X)
		else:
			raise Warning("Unsupported dimension: {}".format(X.shape[-1]))
		pcl.save(P, output, binary=True)
	else:
		raise Warning("Unsupported format: {}".format(format))
	return output


def load(filename, shape=None, dtype=None):
	format = path.splitext(filename)[-1]
	if '.bin' in format:
		assert shape
		assert dtype
		return np.fromfile(filename, dtype=dtype).reshape(*shape)
	elif '.npy' in format:
		X = np.load(filename)
	elif pcl and '.ply' in format or '.pcd' in format:
		X = pcl.load(filename)
		X = np.asarray(X)
	else:
		raise Warning("Unsupported format: {}".format(format))
	
	if shape is not None:
		X = X.reshape(*shape)
	if dtype is not None:
		X = X.astype(dtype)
	return X