import sys, os, stat
from string import Template

import numpy as np

from PuzzleLib.Compiler.Codegen.Types import PointerType
from PuzzleLib.Compiler.Codegen.Types import half_t, half2_t, float_t, double_t
from PuzzleLib.Compiler.Codegen.Types import schar_t, short_t, short2_t, int_t, llong_t
from PuzzleLib.Compiler.Codegen.Types import uchar_t, ushort_t, ushort2_t, uint_t, ullong_t


dtypeToCtype = {
	np.float16: half_t,
	np.float32: float_t,
	np.float64: double_t,

	np.int8: schar_t,
	np.int16: short_t,
	np.int32: int_t,
	np.int64: llong_t,

	np.uint8: uchar_t,
	np.uint16: ushort_t,
	np.uint32: uint_t,
	np.uint64: ullong_t
}


ctypeToDtype = {ctype: dtype for (dtype, ctype) in dtypeToCtype.items()}


class SourceModule:
	Driver = None


	def __init__(self, source, options=None, includes=None, externC=False, verbose=True, debug=False, name=None):
		self.source = source
		self.externC = externC

		self.options = options if options is not None else self.getDefaultOptions()
		self.includes = includes

		if debug and name is None:
			raise self.Driver.RtcError("invalid source module name for debug mode")

		self.debug = debug
		self.name = name

		self.verbose = verbose
		self.cumod = None

		self.functions = {}


	def build(self):
		source = "extern \"C\"\n{\n%s\n}\n" % self.source if self.externC else self.source

		options = self.options
		name = "%s.debug.cu" % self.name if self.debug else (None if self.name is None else "%s.cu" % self.name)

		if self.debug:
			with open(name, mode="w") as f:
				f.write(source)

			os.chmod(name, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
			options = options + ["-G"]

		ptx, log = self.Driver.compile(source, options=options, includes=self.includes, name=name)

		if ptx is None:
			text = log if self.debug else "%s\nSource:\n%s" % (
				log,
				"\n".join("%-4s    %s" % (i + 1, line) for i, line in enumerate(source.splitlines(keepends=False)))
			)

			raise self.Driver.RtcError(text)

		elif log is not None and self.verbose:
			print(log, flush=True)

		self.cumod = self.Driver.Module(ptx)


	def getFunction(self, name):
		func = self.functions.get(name, None)

		if func is None:
			if self.cumod is None:
				self.build()

			func = self.cumod.getFunction(name)
			self.functions[name] = func

		return func


	def __len__(self):
		assert False


	def __getattr__(self, name):
		return self.getFunction(name)


	@classmethod
	def getDefaultOptions(cls):
		deviceIdx = cls.Driver.Device.getCurrent()

		return [
			"-arch=compute_%s%s" % cls.Driver.Device(deviceIdx).computeCapability(), "-use_fast_math",
			"-I%s%sinclude" % (os.environ["CUDA_PATH"] if sys.platform == "win32" else "/usr/local/cuda", os.sep)
		]


class Kernel:
	Driver = None
	SourceModule = None

	warpBit, warpSize = None, None
	blockBit, blockSize = None, None


	def __init__(self, arguments, name):
		self.arguments, self.name = arguments, name
		self.module = None


	def prepareArguments(self, args):
		GPUArray = self.Driver.GPUArray
		size = next(arg.size for arg in args if isinstance(arg, GPUArray))

		args = tuple(
			ctypeToDtype[T](arg) if not isinstance(T, PointerType) else arg
			for arg, (T, name) in zip(args, self.arguments)
		)

		return size, args


	def generateSource(self):
		raise NotImplementedError()


class ElementwiseKernel(Kernel):
	def __init__(self, arguments, operation, name, preambule=""):
		super().__init__(arguments, name)

		self.operation = operation
		self.preambule = preambule


	def generateSource(self):
		arguments = ", ".join(
			(T.restrict if isinstance(T, PointerType) else T).typegen(asDecl=True) % name
			for T, name in self.arguments
		)

		return self.eltwiseTmpl.substitute(
			arguments=arguments, operation=self.operation, name=self.name, preambule=self.preambule
		)


	def prepareForSlice(self, slc, size, args):
		funcname = "%s_strided" % self.name

		start = 0 if slc.start is None else slc.start
		stop = size if slc.stop is None else slc.stop
		step = 1 if slc.step is None else slc.step

		args += (np.int32(start), np.int32(stop), np.int32(step))
		size = (stop - start + step - 1) // step

		return funcname, size, args


	@classmethod
	def prepareGrid(cls, size):
		if size < cls.blockSize:
			block, grid = ((size + cls.warpSize - 1) >> cls.warpBit << cls.warpBit, 1, 1), (1, 1, 1)
		else:
			block, grid = (cls.blockSize, 1, 1), ((size + cls.blockSize - 1) >> cls.blockBit, 1, 1)

		return block, grid


	def __call__(self, *args, **kwargs):
		if self.module is None:
			self.module = self.SourceModule(self.generateSource(), name=self.name)

		funcname = self.name
		size, args = self.prepareArguments(args)

		slc = kwargs.get("slice", None)
		if slc is not None:
			funcname, size, args = self.prepareForSlice(slc, size, args)

		func = self.module.getFunction(funcname)

		block, grid = self.prepareGrid(size)
		func(*args, np.intp(size), block=block, grid=grid)


	eltwiseTmpl = Template("""

$preambule

extern "C" __global__ void $name($arguments, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
	{
		$operation;
	}
}


extern "C" __global__ void ${name}_strided($arguments, int start, int stop, int step)
{
	int i = start + (threadIdx.x + blockIdx.x * blockDim.x) * step;
	if (i < stop)
	{
		$operation;
	}
}

""")


class ElementHalf2Kernel(ElementwiseKernel):
	def __init__(self, arguments, operation2, operation, name, preambule=""):
		super().__init__(arguments, operation, name, preambule)
		self.operation2 = operation2


	@classmethod
	def prepareGrid(cls, size):
		return super().prepareGrid(size >> 1 if size > 1 else 1)


	def generateSource(self):
		args = tuple((T.restrict if isinstance(T, PointerType) else T, name) for T, name in self.arguments)
		arguments = ", ".join(T.typegen(asDecl=True) % name for T, name in args)

		casts = []

		for T, name in args:
			U = T.unqual

			if isinstance(U, PointerType):
				B = U.basetype.unqual

				extB = {
					half_t: half2_t,
					short_t: short2_t,
					ushort_t: ushort2_t
				}.get(B, None)

				if extB is not None:
					casts.append(
						"%s %s2 __attribute__((unused)) = (%s)%s;" % (T.basedWith(extB), name, U.basedWith(extB), name)
					)

		return self.eltwiseTmpl.substitute(
			name=self.name, arguments=arguments, casts="\n".join(casts),
			operation2=self.operation2, operation=self.operation, preambule=self.preambule
		)


	def prepareForSlice(self, slc, size, args):
		assert False


	eltwiseTmpl = Template("""

$preambule
#include <cuda_fp16.h>


extern "C" __global__ void $name($arguments, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size / 2)
	{
		$casts
		$operation2;
	}

	if (size % 2 == 1 && i == size / 2 - 1 || size == 1 && i == 0)
	{
		i = size - 1;
		$operation;
	}
}

""")


class ReductionKernel(Kernel):
	def __init__(self, outtype, neutral, reduceExpr, mapExpr, arguments, name):
		super().__init__(arguments, name)

		self.outtype, self.neutral = outtype, neutral
		self.reduceExpr, self.mapExpr = reduceExpr, mapExpr


	def generateSource(self):
		T = dtypeToCtype[self.outtype]

		arguments = [(T.restrict if isinstance(T, PointerType) else T, name) for T, name in self.arguments]
		arguments = ", ".join(T.typegen(asDecl=True) % name for T, name in arguments)

		stage1 = self.reduceTmpl.substitute(
			T=T, neutral=self.neutral, reduceExpr=self.reduceExpr, mapExpr=self.mapExpr,
			arguments=arguments, warpSize=self.warpSize, NT=self.blockSize, name="%s_stage1" % self.name
		)

		stage2 = self.reduceTmpl.substitute(
			T=T, neutral=self.neutral, reduceExpr=self.reduceExpr, mapExpr="indata[i]",
			arguments="const %s* indata" % T, warpSize=self.warpSize, NT=self.blockSize, name="%s_stage2" % self.name
		)

		return stage1 + stage2


	def reduce(self, stage, allocator, *args):
		size, args = self.prepareArguments(args)

		blocks = min((size + self.blockSize - 1) >> self.blockBit, self.blockSize)
		partials = self.Driver.GPUArray.empty((blocks, ) if blocks > 1 else (), dtype=self.outtype, allocator=allocator)

		kernel = self.module.getFunction("%s_stage%s" % (self.name, stage))
		kernel(*args, partials, np.int32(size), block=(self.blockSize, 1, 1), grid=(blocks, 1, 1))

		return self.reduce(2, allocator, partials) if blocks > 1 else partials


	def __call__(self, *args, **kwargs):
		if self.module is None:
			self.module = self.SourceModule(self.generateSource(), name=self.name)

		allocator = kwargs.get("allocator", None)
		return self.reduce(1, allocator, *args)


	reduceTmpl = Template("""

#undef READ_AND_MAP
#undef REDUCE

#define READ_AND_MAP(i) ($mapExpr)
#define REDUCE(a, b) ($reduceExpr)


extern "C" __global__ void $name($arguments, $T *partials, int size)
{
	__shared__ $T sdata[$warpSize];

	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * $NT;

	$T acc = $neutral;

	for (int i = gid; i < size; i += $NT * gridDim.x)
		acc = REDUCE(acc, READ_AND_MAP(i));

	for (int mask = $warpSize / 2; mask > 0; mask /= 2)
	{
		$T upval = __shfl_xor_sync((unsigned)-1, acc, mask, $warpSize);
		acc = REDUCE(acc, upval);
	}

	if (tid % $warpSize == 0)
		sdata[tid / $warpSize] = acc;

	__syncthreads();
	int nwarps = $NT / $warpSize;

	if (tid < $warpSize)
	{
		acc = (tid < nwarps) ? sdata[tid] : $neutral;

		for (int mask = $warpSize / 2; mask > 0; mask /= 2)
		{
			$T upval = __shfl_xor_sync((unsigned)-1, acc, mask, $warpSize);
			acc = REDUCE(acc, upval);
		}
	}

	if (tid == 0)
		partials[blockIdx.x] = acc;
}

""")


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx)

		rtcTest(bnd)
		eltwiseTest(bnd)
		reductionTest(bnd)


def rtcTest(bnd):
	source = """

extern "C" __global__ void linearKer(float *outdata, const float *indata, float a, float b, int size)
{
	int tid = threadIdx.x;
	int gridsize = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x;

	for (int i = start + tid; i < size; i += gridsize)
		outdata[i] = a * indata[i] + b;
}

"""

	options = bnd.SourceModule.getDefaultOptions()
	ptx, errors = bnd.Driver.compile(source, options=["-lineinfo"] + options, name="linearKer.c")

	assert ptx is not None and errors is None
	print(ptx.decode())


def eltwiseTest(bnd):
	hostInData = np.random.randint(0, 1000, size=(1 << 18, ), dtype=np.int32)

	indata = bnd.GPUArray.toGpu(hostInData)
	outdata = bnd.GPUArray.empty((1 << 18, ), dtype=np.int32)

	square = bnd.ElementwiseKernel(
		[(int_t.ptr, "outdata"), (int_t.const.ptr, "indata")],
		"outdata[i] = indata[i] * indata[i]",
		"square"
	)

	square(outdata, indata)

	hostOutData = hostInData**2
	assert np.allclose(hostOutData, outdata.get())

	square(outdata, outdata, slice=slice(None, None, 10))

	hostOutData[::10] = hostOutData[::10]**2
	assert np.allclose(hostOutData, outdata.get())


def reductionTest(bnd):
	sumkernel = bnd.ReductionKernel(
		np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="data[i]", arguments=[(float_t.const.ptr, "data")],
		name="sum"
	)

	hostData1 = np.random.randn((1 << 18) + 1).astype(np.float32)
	hostData2 = np.ones(shape=((1 << 20) + 1, ), dtype=np.float32)

	for hostData in (hostData1, hostData2):
		data = bnd.GPUArray.toGpu(hostData)

		acc = sumkernel(data)
		hostAcc = np.sum(hostData)

		assert np.isclose(hostAcc, acc.get())


if __name__ == "__main__":
	unittest()
