import sys, os

import numpy as np

from PuzzleLib.Compiler.Toolchain import guessToolchain
from PuzzleLib.Compiler.Containers.Vector.Generate import generateVector
from PuzzleLib.Compiler.Malloc.Generate import generateMalloc


def buildDriver(debugmode, verbose):
	cc = prepareCompiler(debugmode, verbose)
	cc = prepareCuda(cc)

	generateTemplates(path=".")

	driver = "../Driver" + ("_d" if debugmode >= 3 else "") + cc.pydext
	cc.build(driver, collectSources(path=".")).clearPath("..")

	return driver


def prepareCompiler(debugmode, verbose):
	cc = guessToolchain(verbose=verbose)
	cc.addLibrary("numpy", [np.get_include()], [], [])

	if debugmode > 0:
		optlevel, debuglevel = 0, 3
		cc.addDefine("ENABLE_TRACE_MALLOC")

		if debugmode >= 2:
			cc.addDefine("TRACE_CUDA_DRIVER", "TRACE_CUDA_CURAND", "TRACE_CUDA_CUBLAS", "TRACE_CUDA_CUDNN")

			if debugmode >= 3:
				cc.libraries[0] += "_d"

				if sys.platform == "win32":
					cc.addDefine("_DEBUG")

	else:
		optlevel, debuglevel = 4, 0
		cc.addDefine("NDEBUG")

	return cc.withOptimizationLevel(level=optlevel, debuglevel=debuglevel)


def prepareCuda(cc):
	if sys.platform == "win32":
		CUDA_PATH = os.environ["CUDA_PATH"]
		include, lib = os.path.join(CUDA_PATH, "include"), os.path.join(CUDA_PATH, "lib\\x64")

	else:
		include, lib = "/usr/local/cuda/include", "/usr/local/cuda/lib64"

	return cc.addLibrary("cuda", [include], [lib], ["cuda", "cudart", "nvrtc", "curand", "cublas", "cudnn"])


def generateTemplates(path):
	generateVector(
		name="Cuda_AllocVector", T="Cuda_Ptr", minCapacity=128,
		destruct="CUDA_FREE",
		headerPreambule="""
#include "Common.h"


typedef void *Cuda_Ptr;
""",
		bodyPreambule="""
#include "../TraceMalloc/TraceMalloc.gen.h"
#define CUDA_FREE(ptr) CUDA_ASSERT(cudaFree(ptr))
""",
		malloc="TRACE_MALLOC", free="TRACE_FREE", filename=os.path.join(path, "Core/AllocVector")
	)

	generateMalloc(name="TraceMalloc", filename=os.path.join(path, "TraceMalloc/TraceMalloc"))


def collectSources(path):
	return collectCoreSources(path) + collectLibSources(path) + collectDnnSources(path)


def collectCoreSources(path):
	sources = [
		"./Core/AllocVector.gen.c",
		"./Core/Allocator.c",

		"./Core/Array.c",
		"./Core/Buffer.c",
		"./Core/Device.c",
		"./Core/Driver.c",
		"./Core/Module.c",
		"./Core/Stream.c",

		"./TraceMalloc/AllocTree.gen.c",
		"./TraceMalloc/TraceMalloc.gen.c"
	]

	return [os.path.join(path, source) for source in sources]


def collectLibSources(path):
	sources = [
		"./Libs/CuRand.c",
		"./Libs/CuBlas.c"
	]

	return [os.path.join(path, source) for source in sources]


def collectDnnSources(path):
	sources = [
		"./Libs/CuDnn.c",
		"./Libs/CuDnnPool.c",
		"./Libs/CuDnnNorm.c",
		"./Libs/CuDnnMemory.c",
		"./Libs/CuDnnRnn.c",
		"./Libs/CuDnnSpatialTf.c"
	]

	return [os.path.join(path, source) for source in sources]


def main():
	return buildDriver(debugmode=0, verbose=2)


if __name__ == "__main__":
	main()
