import numpy as np


class CPUArray:
	def __init__(self, shape, dtype, data=None, acquire=False):
		if isinstance(shape, int):
			shape = (shape, )

		self.shape = tuple(shape)
		self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)

		if data is not None:
			assert shape == data.shape and dtype == data.dtype

			if not isinstance(data, np.ndarray):
				data = np.array(data)

			self.data = data if acquire else np.ascontiguousarray(np.copy(data))

		else:
			self.data = np.empty(shape, dtype=dtype)


	@property
	def flags(self):
		return self.data.flags


	@property
	def strides(self):
		return self.data.strides


	@property
	def size(self):
		return self.data.size


	@property
	def ndim(self):
		return self.data.ndim


	@property
	def nbytes(self):
		return self.data.nbytes


	@property
	def ptr(self):
		return self.data.__array_interface__["data"][0]


	def get(self, copy=True):
		return np.copy(self.data) if copy else self.data


	def set(self, ary):
		if isinstance(ary, CPUArray):
			ary = ary.data

		self.data[...] = ary


	def fill(self, value):
		self.data[...] = value


	def reshape(self, *args):
		data = self.data.reshape(*args)
		return CPUArray(data.shape, data.dtype, data=data, acquire=True)


	def ravel(self):
		data = self.data.reshape((self.data.size, ))
		return CPUArray(data.shape, data.dtype, data=data, acquire=True)


	def view(self, dtype):
		data = self.data.view(dtype)
		return CPUArray(data.shape, data.dtype, data=data, acquire=True)


	def copy(self):
		data = self.data.copy()
		return CPUArray(data.shape, data.dtype, data=data, acquire=True)


	@staticmethod
	def unpackArg(arg):
		return arg if isinstance(arg, (int, float)) else arg.data


	def __setitem__(self, item, other):
		self.data[item] = other.data


	def __getitem__(self, item):
		data = self.data.__getitem__(item)
		return CPUArray(data.shape, data.dtype, data=data, acquire=True)


	def __add__(self, other):
		result = self.data.__add__(self.unpackArg(other))
		return CPUArray(result.shape, result.dtype, data=result, acquire=True)


	def __radd__(self, other):
		return self.__add__(other)


	def __iadd__(self, other):
		self.data.__iadd__(self.unpackArg(other))
		return self


	def __sub__(self, other):
		result = self.data.__sub__(self.unpackArg(other))
		return CPUArray(result.shape, result.dtype, data=result, acquire=True)


	def __isub__(self, other):
		self.data.__isub__(self.unpackArg(other))
		return self


	def __mul__(self, other):
		result = self.data.__mul__(self.unpackArg(other))
		return CPUArray(result.shape, result.dtype, data=result, acquire=True)


	def __rmul__(self, other):
		return self.__mul__(other)


	def __imul__(self, other):
		self.data.__imul__(self.unpackArg(other))
		return self


	def __truediv__(self, other):
		result = self.data.__truediv__(self.unpackArg(other))
		return CPUArray(result.shape, result.dtype, data=result, acquire=True)


	def __itruediv__(self, other):
		self.data.__itruediv__(self.unpackArg(other))
		return self


	def __str__(self):
		return str(self.data)


	def __repr__(self):
		return repr(self.data)


	@staticmethod
	def toDevice(ary, **_):
		return CPUArray(ary.shape, ary.dtype, data=ary)


	@staticmethod
	def empty(shape, dtype, **_):
		return CPUArray(shape, dtype)


	@staticmethod
	def zeros(shape, dtype, **_):
		ary = np.zeros(shape, dtype=dtype)
		return CPUArray(shape, ary.dtype, data=ary, acquire=True)


	@staticmethod
	def minimum(ary):
		mn = np.min(ary.data)
		return CPUArray(mn.shape, mn.dtype, data=mn, acquire=True)


	@staticmethod
	def maximum(ary):
		mx = np.max(ary.data)
		return CPUArray(mx.shape, mx.dtype, data=mx, acquire=True)


	@staticmethod
	def arange(start=None, stop=None, step=None, dtype=None):
		ary = np.arange(start, stop, step, dtype)
		return CPUArray(ary.shape, ary.dtype, data=ary, acquire=True)


	@staticmethod
	def full(shape, fillvalue, dtype):
		ary = np.full(shape, fillvalue, dtype)
		return CPUArray(ary.shape, ary.dtype, data=ary, acquire=True)


	@staticmethod
	def moveaxis(ary, src, dest):
		out = np.ascontiguousarray(np.moveaxis(ary.data, src, dest))
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)


	@staticmethod
	def swapaxes(ary, axis1, axis2):
		out = np.copy(np.swapaxes(ary.data, axis1, axis2))
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)
