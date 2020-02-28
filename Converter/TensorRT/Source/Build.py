import sys, os

import pybind11

from PuzzleLib.Compiler.Toolchain import guessToolchain, guessNVCCToolchain
from PuzzleLib.Compiler.BuildSystem import Rule, LinkRule, build


def buildDriver():
	cc, nvcc = prepareCompilers()
	rules, linkrule = createRules(cc, nvcc)

	build(rules, linkrule)
	cc.clearPath("..")


def prepareCompilers():
	cc = guessToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0).cppMode(True)
	nvcc = guessNVCCToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0)

	if sys.platform == "linux":
		cc.includeDirs.append(pybind11.get_include(user=True))

		cc.addLibrary(
			"tensorrt",
			["/usr/local/cuda/include", "/usr/local/include/python%s.%s" % sys.version_info[:2]],
			["/usr/local/cuda/lib64"],
			["cudart", "nvinfer", "nvinfer_plugin", "nvcaffe_parser", "nvonnxparser"]
		)

	elif sys.platform == "win32":
		CUDA_PATH = os.environ["CUDA_PATH"]

		cc.addLibrary(
			"tensorrt",
			[os.path.join(CUDA_PATH, "include")],
			[os.path.join(CUDA_PATH, "lib/x64")],
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
	buildDriver()


if __name__ == "__main__":
	main()
