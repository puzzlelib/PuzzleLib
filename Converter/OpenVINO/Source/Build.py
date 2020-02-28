import sys, os
import pybind11

from PuzzleLib.Compiler.Toolchain import guessToolchain


def buildDriver():
	cc = prepareCompiler()

	cc.build(extfile="../Driver%s" % cc.pydext, sourcefiles="./Driver.cpp")
	cc.clearPath("..")


def prepareCompiler():
	cc = guessToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0).cppMode(True)

	if sys.platform == "linux":
		cc.includeDirs.append(pybind11.get_include(user=True))

		OPENVINO_PATH = os.path.expanduser("~/intel/openvino")

		cc.addLibrary(
			"openvino",
			[os.path.join(OPENVINO_PATH, "./inference_engine/include")],
			[os.path.join(OPENVINO_PATH, "./inference_engine/lib/intel64")],
			["inference_engine"]
		)

	elif sys.platform == "win32":
		OPENVINO_PATH = "C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379"

		cc.addLibrary(
			"openvino",
			[os.path.join(OPENVINO_PATH, "./inference_engine/include")],
			[os.path.join(OPENVINO_PATH, "./inference_engine/lib/intel64/Release")],
			["inference_engine"]
		)

	else:
		raise NotImplementedError(sys.platform)

	return cc


def main():
	buildDriver()


if __name__ == "__main__":
	main()
