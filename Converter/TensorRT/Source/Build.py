import sys, os

import pybind11

from PuzzleLib.Compiler.Toolchain import guessToolchain, guessNVCCToolchain
from PuzzleLib.Compiler.BuildSystem import Rule, LinkRule, build


def buildDriver():
	cc, nvcc = prepareCompilers()
	rules, linkrule = createRules(cc, nvcc)

	build(rules, linkrule)
	cc.clearPath("..")

	return linkrule.target


def findLibraryPath():
	TRT_PATH = os.environ.get("TRT_PATH", None)

	if TRT_PATH is None:
		if sys.platform == "linux":
			TRT_PATH = "/usr/local/cuda"

		elif sys.platform == "win32":
			TRT_PATH = os.environ["CUDA_PATH"]

		else:
			raise NotImplementedError(sys.platform)

	return TRT_PATH


def prepareCompilers():
	cc = guessToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0).cppMode(True)
	nvcc = guessNVCCToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0)

	TRT_PATH = findLibraryPath()

	if sys.platform == "linux":
		cc.includeDirs.append(pybind11.get_include(user=True))

		cc.addLibrary(
			"tensorrt",
			[os.path.join(TRT_PATH, "include"), "/usr/local/include/python%s.%s" % sys.version_info[:2]],
			[os.path.join(TRT_PATH, "lib64")],
			["cudart", "nvinfer", "nvinfer_plugin", "nvcaffe_parser", "nvonnxparser"]
		)

	elif sys.platform == "win32":
		cc.addLibrary(
			"tensorrt",
			[os.path.join(TRT_PATH, "include")],
			[os.path.join(TRT_PATH, "lib/x64")],
			["cudart", "nvinfer", "nvinfer_plugin", "nvparsers", "nvonnxparser"]
		)

	else:
		raise NotImplementedError(sys.platform)

	return cc, nvcc


def createRules(cc, nvcc):
	rules = [
		Rule(target="./PRelu%s" % nvcc.oext, deps=[
			"./Plugins.h",
			"./PRelu.h",
			"./PRelu.cu"
		], toolchain=nvcc),

		Rule(target="./ReflectPad1D%s" % nvcc.oext, deps=[
			"./Plugins.h",
			"./ReflectPad1D.h",
			"./ReflectPad1D.cu"
		], toolchain=nvcc),

		Rule(target="./Plugins%s" % cc.oext, deps=[
			"./Plugins.h",
			"./PRelu.h",
			"./ReflectPad1D.h",
			"./Plugins.cpp"
		], toolchain=cc),

		Rule(target="./Driver%s" % cc.oext, deps=[
			"./Plugins.h",
			"./Driver.cpp"
		], toolchain=cc)
	]

	linkrule = LinkRule(target="../Driver%s" % cc.pydext, deps=rules, toolchain=cc)
	return rules, linkrule


def main():
	return buildDriver()


if __name__ == "__main__":
	main()
