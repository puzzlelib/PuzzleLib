import sys, os
import pybind11

from PuzzleLib.Compiler.Toolchain import guessToolchain


def buildDriver():
	cc = prepareCompiler()

	driver = "../Driver%s" % cc.pydext
	cc.build(driver, sourcefiles="./Driver.cpp").clearPath("..")

	return driver


def findLibraryPath():
	OPENVINO_PATH = os.environ.get("OPENVINO_PATH", None)

	if OPENVINO_PATH is None:
		if sys.platform == "linux":
			OPENVINO_PATH = os.path.expanduser("~/intel/openvino")

		elif sys.platform == "win32":
			raise OSError("OpenVINO path needs to be specified in the system variables as OPENVINO_PATH")

		else:
			raise NotImplementedError(sys.platform)

	return OPENVINO_PATH


def prepareCompiler():
	cc = guessToolchain(verbose=2).withOptimizationLevel(level=4, debuglevel=0).cppMode(True)
	OPENVINO_PATH = findLibraryPath()

	if sys.platform == "linux":
		cc.includeDirs.append(pybind11.get_include(user=True))

		cc.addLibrary(
			"openvino",
			[os.path.join(OPENVINO_PATH, "inference_engine/include")],
			[os.path.join(OPENVINO_PATH, "inference_engine/lib/intel64")],
			["inference_engine"]
		)

	elif sys.platform == "win32":
		cc.addLibrary(
			"openvino",
			[os.path.join(OPENVINO_PATH, "inference_engine/include")],
			[os.path.join(OPENVINO_PATH, "inference_engine/lib/intel64/Release")],
			["inference_engine"]
		)

	else:
		raise NotImplementedError(sys.platform)

	return cc


def main():
	return buildDriver()


if __name__ == "__main__":
	main()
