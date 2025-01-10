'''
Helper functions for this project.

Author: Gerald Baulig
'''

## Build in
from __future__ import print_function
from sys import version_info
from time import time


__python2__ = version_info[0] == 2


if __python2__:
	import glob as _glob
	
	def glob(wc, recursive=False):
		return _glob.glob(wc)
	
	def iglob(wc, recursive=False):
		return _glob.iglob(wc)	
else:
	from glob import glob, iglob


def log(*nargs, **kwargs):
	if log.verbose:
		log.func(*nargs, **kwargs)
log.verbose = False
log.func = lambda *nargs, **kwargs: print(*nargs, **kwargs)


def myinput(prompt, default=None, cast=None):
	''' myinput(prompt, default=None, cast=None) -> arg
	Handle an interactive user input.
	Returns a default value if no input is given.
	Casts or parses the input immediately.
	Loops the input prompt until a valid input is given.
	
	Args:
		prompt: The prompt or help text.
		default: The default value if no input is given.
		cast: A cast or parser function.
	'''
	while True:
		arg = input(prompt)
		if arg == '':
			return default
		elif cast != None:
			try:
				return cast(arg)
			except:
				print("Invalid input type. Try again...")
		else:
			return arg
	pass


def ifile(wildcards, sort=False, recursive=True):
	def sglob(wc):
		if sort:
			return sorted(glob(wc, recursive=recursive))
		else:
			return iglob(wc, recursive=recursive)

	if isinstance(wildcards, str):
		for wc in sglob(wildcards):
			yield wc
	elif isinstance(wildcards, list):
		if sort:
			wildcards = sorted(wildcards)
		for wc in wildcards:
			if any((c in '*?[') for c in wc):
				for c in sglob(wc):
					yield c
			else:
				yield wc
	else:
		raise TypeError("wildecards must be string or list.")


def time_delta(start=None):
	''' time_delta() -> delta
	Captures time delta from last call.
	
	Yields:
		delta: Past time in seconds.
	'''
	if not start:
		start = time()
	
	while True:
		curr = time()
		delta = curr - start
		start = curr
		yield delta


class Prototype():
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			self.__dict__[k] = v
	pass

	def __bool__(self):
		return True
	
	def __iter__(self):
		return iter(self.__dict__)
