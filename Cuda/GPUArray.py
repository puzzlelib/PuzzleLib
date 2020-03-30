import numpy as np


def memoize(fn):
	cache = {}

	def memoizer(*args):
		obj = cache.get(args, None)
		if obj is not None:
			return obj

		obj = fn(*args)
		cache[args] = obj

		return obj

	return memoizer


def extendGPUArray(Driver, ElementwiseKernel, ElementHalf2Kernel, ReductionKernel):
	from PuzzleLib.Cuda.SourceModule import dtypeToCtype
	GPUArray = Driver.GPUArray


	@memoize
	def getFillKernel(dtype):
		ctype = dtypeToCtype[dtype.type]

		arguments = [(ctype.ptr, "data"), (ctype, "value")]
		name = "fillKer"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"data2[i] = half2(value, value);",
				"data[i] = value",
				name
			)

		else:
			return ElementwiseKernel(
				arguments,
				"data[i] = value",
				name
			)


	@memoize
	def getConvertTypeKernel(intype, outtype):
		assert intype != outtype
		inctype, outctype = dtypeToCtype[intype.type], dtypeToCtype[outtype.type]

		arguments = [(outctype.ptr, "outdata"), (inctype.const.ptr, "indata")]
		name = "convertTypeKer"

		if intype == np.float16 or outtype == np.float16:
			if intype == np.float16:
				assert outtype == np.float32
				operation2 = "((float2 *)outdata)[i] = __half22float2(indata2[i])"
				operation = "outdata[i] = indata[i]"

			else:
				assert intype == np.float32
				operation2 = "outdata2[i] = __float22half2_rn(((float2 *)indata)[i])"
				operation = "outdata[i] = indata[i]"

			return ElementHalf2Kernel(arguments, operation2, operation, name)

		else:
			return ElementwiseKernel(
				arguments,
				"outdata[i] = (%s)indata[i]" % outctype,
				name
			)


	@memoize
	def getMinMaxKernel(dtype):
		ctype = dtypeToCtype[dtype.type]

		if issubclass(dtype.type, np.floating):
			typeinfo = np.finfo(dtype)
		elif issubclass(dtype.type, np.integer):
			typeinfo = np.iinfo(dtype)
		else:
			raise NotImplementedError(dtype)

		minval, maxval = typeinfo.min, typeinfo.max
		arguments = [(ctype.const.ptr, "data")]

		minKer = ReductionKernel(
			outtype=dtype.type, neutral=maxval, reduceExpr="min(a, b)", mapExpr="data[i]",
			arguments=arguments, name="min"
		)

		maxKer = ReductionKernel(
			outtype=dtype.type, neutral=minval, reduceExpr="max(a, b)", mapExpr="data[i]",
			arguments=arguments, name="max"
		)

		return minKer, maxKer


	@memoize
	def getArithmKernel(op, dtype):
		ctype = dtypeToCtype[dtype.type]

		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "lhs"), (ctype.const.ptr, "rhs")]
		name = "arithmKer"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 lhsvec = __half22float2(lhs2[i]), rhsvec = __half22float2(rhs2[i]);
				outdata2[i] = __float22half2_rn(make_float2(lhsvec.x %s rhsvec.x, lhsvec.y %s rhsvec.y));
				""" % (op, op),
				"outdata[i] = (float)lhs[i] %s (float)rhs[i]" % op,
				name
			)

		else:
			return ElementwiseKernel(
				arguments,
				"outdata[i] = lhs[i] %s rhs[i]" % op,
				name
			)


	@memoize
	def getInplaceArithmKernel(op, dtype):
		ctype = dtypeToCtype[dtype.type]

		return ElementwiseKernel(
			[(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")],
			"outdata[i] %s indata[i]" % op,
			"inplaceArithmKer"
		)


	def enforceContiguous(self):
		if not self.contiguous:
			raise ValueError("gpuarray is not contiguous")


	def findParentAllocator(*args):
		for arg in args:
			parent = arg.parent

			if isinstance(parent, Driver.MemoryPool):
				return parent

		return None


	def enforceEqualShapesAndDtypes(self, other):
		enforceContiguous(self)
		enforceContiguous(other)

		if self.shape != other.shape:
			raise ValueError("gpuarray shapes are not equal")

		if self.dtype != other.dtype:
			raise ValueError("gpuarray datatypes are not equal")


	def fill(self, val):
		enforceContiguous(self)
		item = self.dtype.type(val)

		if self.dtype.itemsize == 4:
			self.gpudata[:self.nbytes].fillD32(item.view(np.uint32))
		elif self.dtype.itemsize == 2:
			self.gpudata[:self.nbytes].fillD16(item.view(np.uint16))
		elif self.dtype.itemsize == 1:
			self.gpudata[:self.nbytes].fillD8(item.view(np.uint8))
		else:
			getFillKernel(self.dtype)(self, self.dtype.type(val))

		return self


	def astype(self, dtype):
		enforceContiguous(self)

		allocator = findParentAllocator(self.gpudata)
		outdata = GPUArray.empty(self.shape, dtype, allocator=allocator)

		if self.dtype == dtype:
			outdata.set(self)
		else:
			getConvertTypeKernel(self.dtype, np.dtype(dtype))(outdata, self)

		return outdata


	def setitem(self, key, value):
		self[key].set(value)


	def aryMin(self):
		enforceContiguous(self)

		minKer, _ = getMinMaxKernel(self.dtype)
		return minKer(self)


	def aryMax(self):
		enforceContiguous(self)

		_, maxKer = getMinMaxKernel(self.dtype)
		return maxKer(self)


	def add(self, other):
		enforceEqualShapesAndDtypes(self, other)

		allocator = findParentAllocator(self.gpudata, other.gpudata)
		outdata = GPUArray.empty(self.shape, self.dtype, allocator=allocator)

		getArithmKernel("+", self.dtype)(outdata, self, other)
		return outdata


	def mul(self, other):
		enforceEqualShapesAndDtypes(self, other)

		allocator = findParentAllocator(self.gpudata, other.gpudata)
		outdata = GPUArray.empty(self.shape, self.dtype, allocator=allocator)

		getArithmKernel("*", self.dtype)(outdata, self, other)
		return outdata


	def iadd(self, other):
		enforceEqualShapesAndDtypes(self, other)

		getInplaceArithmKernel("+=", self.dtype)(self, other)
		return self


	def imul(self, other):
		enforceEqualShapesAndDtypes(self, other)

		getInplaceArithmKernel("*=", self.dtype)(self, other)
		return self


	GPUArray.__setitem__ = setitem

	GPUArray.fill = fill
	GPUArray.astype = astype
	GPUArray.min = aryMin
	GPUArray.max = aryMax

	GPUArray.__add__ = add
	GPUArray.__mul__ = mul
	GPUArray.__iadd__ = iadd
	GPUArray.__imul__ = imul


	return GPUArray


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx)

		for dtype, _ in bnd.dtypesSupported():
			arithmTest(bnd, dtype)
			memoryTest(bnd, dtype)


def arithmTest(bnd, dtype):
	hostA = np.random.randn(13, 15).astype(dtype)
	hostB = np.random.randn(13, 15).astype(dtype)

	a, b = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)

	c = a + b
	assert np.allclose(hostA + hostB, c.get())

	d = a * b
	assert np.allclose(hostA * hostB, d.get())

	a.fill(3.0)
	assert np.allclose(np.full_like(hostA, fill_value=3.0), a.get())

	c = b.astype(np.float32 if dtype == np.float16 else np.float16)
	assert np.allclose(hostB.astype(c.dtype), c.get())


def memoryTest(bnd, dtype):
	hostA = np.random.randn(10, 10).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	b = a[:, :6]
	hostB = hostA[:, :6]

	assert np.allclose(hostB.reshape((2, 5, 6)), b.reshape(2, 5, 6).get())
	assert np.allclose(hostB.reshape((5, 2, 3, 2)), b.reshape(5, 2, 3, 2).get())
	assert np.allclose(hostB.reshape((10, 1, 6)), b.reshape(10, 1, 6).get())

	hostA = np.random.randn(10, 10, 10).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	b = a[:, :, :6]
	assert np.allclose(hostA[:, :, :6], b.get())

	hostB = np.random.randn(*b.shape).astype(dtype)
	b.set(hostB)
	assert np.allclose(hostB, b.get())

	hostB = b.get()
	b = a[:, :6, :6]
	assert np.allclose(hostB[:, :6, :6], b.get())

	hostB = np.random.randn(*b.shape).astype(dtype)
	b.set(hostB)
	assert np.allclose(hostB, b.get())

	hostB = np.random.randn(10, 6, 10).astype(dtype)[:, :, :6]
	b.set(hostB)
	assert np.allclose(hostB, b.get())

	hostB = np.random.randn(10, 10, 6).astype(dtype)[:, :6, :]
	b.set(hostB)
	assert np.allclose(hostB, b.get())


if __name__ == "__main__":
	unittest()
