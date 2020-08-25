import sys, os

import pybind11

from PuzzleLib.Compiler.Toolchain import guessToolchain, guessNVCCToolchain
from PuzzleLib.Compiler.BuildSystem import Rule, LinkRule, build


def buildDriver(debugmode=0):
	cc, nvcc = prepareCompilers(debugmode=debugmode)
	rules, linkrule = createRules(cc, nvcc)

	build(rules, linkrule)
	cc.clearPath("..")

	return linkrule.target


def findLibraryPath():
	if sys.platform == "linux":
		CUDA_PATH = "/usr/local/cuda"

	elif sys.platform == "win32":
		CUDA_PATH = os.environ["CUDA_PATH"]

	else:
		raise NotImplementedError(sys.platform)

	TRT_PATH = os.environ.get("TRT_PATH", None)
	TRT_PATH = CUDA_PATH if TRT_PATH is None else TRT_PATH

	return CUDA_PATH, TRT_PATH


def prepareCompilers(debugmode=0):
	level, debuglevel = (0, 3) if debugmode > 0 else (4, 0)

	cc = guessToolchain(verbose=2).withOptimizationLevel(level=level, debuglevel=debuglevel).cppMode(True)
	nvcc = guessNVCCToolchain(verbose=2).withOptimizationLevel(level=level, debuglevel=debuglevel)

	CUDA_PATH, TRT_PATH = findLibraryPath()

	if sys.platform == "linux":
		cc.includeDirs.extend(
			(pybind11.get_include(user=True), "/usr/local/include/python%s.%s" % sys.version_info[:2])
		)

		cc.addLibrary(
			"tensorrt",
			[os.path.join(TRT_PATH, "include")],
			[os.path.join(TRT_PATH, "lib64")],
			["cudart", "nvinfer", "nvcaffe_parser", "nvonnxparser"]
		)

		cc.addLibrary(
			"cuda",
			[os.path.join(CUDA_PATH, "include")],
			[os.path.join(CUDA_PATH, "lib64")],
			["cudnn"]
		)

	elif sys.platform == "win32":
		cc.addLibrary(
			"tensorrt",
			[os.path.join(TRT_PATH, "include")],
			[os.path.join(TRT_PATH, "lib/x64")],
			["cudart", "nvinfer", "nvparsers", "nvonnxparser"]
		)

		cc.addLibrary(
			"cuda",
			[os.path.join(CUDA_PATH, "include")],
			[os.path.join(CUDA_PATH, "lib/x64")],
			["cudnn"]
		)

	else:
		raise NotImplementedError(sys.platform)

	return cc, nvcc


def createRules(cc, nvcc):
	rules = [
		Rule(target="InstanceNorm2D" + nvcc.oext, deps=[
			"Plugins.h",
			"InstanceNorm2D.cpp"
		], toolchain=cc),

		Rule(target="ReflectPad1D" + nvcc.oext, deps=[
			"Plugins.h",
			"ReflectPad1D.cu"
		], toolchain=nvcc),

		Rule(target="Plugins%s" % cc.oext, deps=[
			"Plugins.h",
			"Plugins.cpp"
		], toolchain=cc),

		Rule(target="Driver%s" % cc.oext, deps=[
			"Plugins.h",
			"Driver.cpp"
		], toolchain=cc)
	]

	linkrule = LinkRule(target="../Driver%s" % cc.pydext, deps=rules, toolchain=cc)
	return rules, linkrule


def main():
	return buildDriver(debugmode=0)


if __name__ == "__main__":
	main()
