import os, tempfile, subprocess
from string import Template

from PuzzleLib.Config import libname, Backend, systemLog
from PuzzleLib.Compiler.JIT import getCacheDir, computeHash, FileLock

from PuzzleLib.Hip import Driver as HipDriver

from PuzzleLib.Cuda.SourceModule import SourceModule, ElementwiseKernel, ElementHalf2Kernel, ReductionKernel
from PuzzleLib.Cuda.SourceModule import eltwiseTest, reductionTest


hipWarpBit, hipBlockBit = 6, 8
hipWarpSize, hipBlockSize = 1 << hipWarpBit, 1 << hipBlockBit


class HipSourceModule(SourceModule):
	Driver = HipDriver

	runtimeHeader = """
#include <hip/hip_runtime.h>

#define __shfl_xor_sync(mask, value, laneMask, ...) __shfl_xor(value, laneMask, __VA_ARGS__)
#define __shfl_up_sync(mask, value, delta, ...) __shfl_up(value, delta, __VA_ARGS__)
"""


	def __init__(self, source, options=None, includes=None, externC=False, verbose=True, debug=False, recompile=False,
				 name=None):
		super().__init__(source, options, includes, externC, verbose, debug, name)

		self.recompile = recompile
		self.includes = [] if self.includes is None else self.includes


	def build(self):
		source = self.source.replace("cuda_fp16.h", "hip/hip_fp16.h")
		source = ("%sextern \"C\"\n{\n%s\n}\n" if self.externC else "%s%s") % (self.runtimeHeader, source)

		cachedir = getCacheDir(os.path.join(libname, Backend.hip.name))

		with FileLock(cachedir):
			try:
				codename = self.tryBuild(source, cachedir)

			except subprocess.CalledProcessError as e:
				log = e.output.decode()
				text = log if self.debug else "%s\nSource:\n%s" % (
					log,
					"\n".join("%-4s    %s" % (i + 1, line) for i, line in enumerate(source.splitlines(keepends=False)))
				)

				raise self.Driver.RtcError(text)

		with open(codename, mode="rb") as f:
			hsaco = f.read()

		self.cumod = self.Driver.Module(hsaco)


	def tryBuild(self, source, cachedir):
		options, includes = self.options, self.includes
		hashsum = computeHash(source, *options, *includes)

		codepath = os.path.join(cachedir, hashsum)
		name, srcext = "module" if self.name is None else self.name, ".hip.cpp"

		codename = os.path.join(codepath, "%s.code" % name)
		sourcename = os.path.join(codepath, "%s%s" % (name, srcext))

		if not os.path.exists(codename) or self.recompile:
			os.makedirs(codepath, exist_ok=True)

			args = ["hipcc", "--genco"] + options + ["-o", codename]
			stderr = subprocess.STDOUT if self.verbose else subprocess.DEVNULL

			if systemLog:
				print(
					"[%s] No cache found for HIP extension '%s', performing compilation ..." %
					(libname, name), flush=True
				)

			if not self.debug:
				f = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=srcext, delete=False)
				try:
					with f:
						f.write(source)

					subprocess.check_output(args + [f.name], stderr=stderr)

				finally:
					os.remove(f.name)

			else:
				with open(sourcename, mode="w", encoding="utf-8") as f:
					f.write(source)

				subprocess.check_output(args + [sourcename], stderr=stderr)

		else:
			if systemLog:
				print(
					"[%s] Found cached compilation for HIP extension '%s', skipping compilation ..." %
					(libname, name), flush=True
				)

		return codename


	@classmethod
	def getDefaultOptions(cls):
		deviceIdx = cls.Driver.Device.getCurrent()
		return ["--targets gfx%s" % cls.Driver.Device(deviceIdx).getArch()]


class HipEltwiseKernel(ElementwiseKernel):
	Driver = HipDriver
	SourceModule = HipSourceModule

	warpBit, warpSize = hipWarpBit, hipWarpSize
	blockBit, blockSize = hipBlockBit, hipBlockSize


class HipEltHalf2Kernel(ElementHalf2Kernel):
	Driver = HipDriver
	SourceModule = HipSourceModule

	warpBit, warpSize = hipWarpBit, hipWarpSize
	blockBit, blockSize = hipBlockBit, hipBlockSize


class HipReductionKernel(ReductionKernel):
	Driver = HipDriver
	SourceModule = HipSourceModule

	warpBit, warpSize = hipWarpBit, hipWarpSize
	blockBit, blockSize = hipBlockBit, hipBlockSize

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
		$T upval = __shfl_xor(acc, mask, $warpSize);
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
			$T upval = __shfl_xor(acc, mask, $warpSize);
			acc = REDUCE(acc, upval);
		}
	}

	if (tid == 0)
		partials[blockIdx.x] = acc;
}

""")


def unittest():
	from PuzzleLib.Hip import Backend

	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx)

		eltwiseTest(bnd)
		reductionTest(bnd)


if __name__ == "__main__":
	unittest()
