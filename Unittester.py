import sys, os, traceback, gc
import importlib.util, random, argparse

from colorama import Fore, Style

from PuzzleLib import Config
Config.systemLog = True

from PuzzleLib.Backend.Utils import setupDebugAllocator
from PuzzleLib.Transformers import Transformer, Generator


if "PYCHARM_HOSTED" not in os.environ:
	import colorama
	colorama.init()


def runModuleTest(mod, path, threshold=20):
	itern = 0
	prevpath = os.getcwd()

	while True:
		os.chdir(path)
		try:
			mod.unittest()

		except Exception as e:
			if isinstance(e, AssertionError):
				traceinfo = traceback.extract_tb(sys.exc_info()[-1])
				_, line, _, text = traceinfo[-1]

				e = "An assert error occurred on line %s in statement:\n'%s'" % (line, text)

			print(Fore.YELLOW + "%s unittest failed on try #%s with reason: %s" % (mod, itern + 1, e) + Style.RESET_ALL)

			if itern < threshold - 1:
				itern += 1
				continue

			else:
				print(Fore.RED + "Threshold limit exceeded" + Style.RESET_ALL)
				return False

		finally:
			os.chdir(prevpath)

		print(Fore.LIGHTGREEN_EX + "%s unittest finished successfully" % mod + Style.RESET_ALL)
		return True


def findAllowedNames(filenames, allowed):
	if allowed is None:
		return filenames

	availSourceFiles = []

	for avail in filenames:
		for allowedfile in allowed:
			if avail.endswith(os.path.normpath(allowedfile)):
				availSourceFiles.append(avail)

	return availSourceFiles


def runUnittests(filenames, systemMods, threshold=20, allowed=None):
	failure, success, ignored = [], [], []
	filenames = findAllowedNames(filenames, allowed)

	print("Setuping debug allocator ...")
	setupDebugAllocator()

	if allowed is None:
		for mod in systemMods + [Transformer, Generator]:
			if hasattr(mod, "unittest"):
				if not runModuleTest(mod, os.path.dirname(mod.__file__), threshold=threshold):
					failure.append(mod.__file__)

	gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

	for filename in filenames:
		spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(filename))[0], filename)
		mod = importlib.util.module_from_spec(spec)

		try:
			spec.loader.exec_module(mod)

			if hasattr(mod, "unittest") and not hasattr(mod, "main"):
				print("%s has unittest. Starting ..." % filename)

				dstContainer = success if runModuleTest(mod, os.path.dirname(filename)) else failure
				dstContainer.append(filename)

				gc.collect()

			else:
				ignored.append(filename)

		except Exception as e:
			traceinfo = traceback.extract_tb(sys.exc_info()[-1])
			_, line, _, _ = traceinfo[-1]

			e = "(line: %s, type: %s): %s" % (line, type(e), e)
			print(Fore.RED + "%s testing failed with reason: %s" % (mod, e) + Style.RESET_ALL)

			failure.append(filename)

	if allowed is not None:
		print("Ran unittests on allowed files: %s" % filenames)

	return failure, success, ignored


def parseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument("--allowed", type=str, nargs="*", default=None)
	parser.add_argument("--exclude", type=str, nargs="*", default=[])
	parser.add_argument("--threshold", type=int, default=20)

	args = parser.parse_args()

	allowed = None if args.allowed is None else set(args.allowed)
	exclude, threshold = set(args.exclude), args.threshold

	return allowed, exclude, threshold


def prepareBackend(exclude):
	exclude.update([
		__file__,
		"./Transformers/Transformer.py", "./Transformers/Generator.py",
		"./SetupTools", "./Converter",
		"./Cuda", "./OpenCL", "./CPU", "./Intel"
	])

	if Config.backend == Config.Backend.cuda:
		exclude, systemMods = prepareCudaBackend(exclude)

	elif Config.backend == Config.Backend.opencl:
		exclude, systemMods = prepareOpenCLBackend(exclude)

	elif Config.backend == Config.Backend.cpu:
		exclude, systemMods = prepareCPUBackend(exclude)

	elif Config.backend == Config.Backend.intel:
		exclude, systemMods = prepareIntelBackend(exclude)

	else:
		raise NotImplementedError(Config.backend)

	return exclude, systemMods


def prepareCudaBackend(exclude):
	from PuzzleLib.Cuda import Utils
	from PuzzleLib.Cuda.Wrappers import CuDnn, CuBlas

	exclude.discard("./Cuda")
	exclude.update(["./Cuda/Utils.py", "./Cuda/Wrappers/CuDnn.py", "./Cuda/Wrappers/CuBlas.py"])

	systemMods = [Utils, CuDnn, CuBlas]
	return exclude, systemMods


def prepareOpenCLBackend(exclude):
	from PuzzleLib.OpenCL import Utils
	from PuzzleLib.OpenCL.Wrappers import MIOpen, CLBlas

	exclude.discard("./OpenCL")
	exclude.update(["./OpenCL/Utils.py", "./OpenCL/Wrappers/MIOpen.py", "./OpenCL/Wrappers/CLBlas.py"])

	exclude.update([
		"./Modules/LCN.py", "./Modules/SpatialTf.py", "./Modules/Slice.py",
		"./Modules/BatchNorm3D.py", "./Modules/Conv3D.py", "./Modules/Deconv3D.py",
		"./Modules/MaxPool3D.py", "./Modules/SubtractMean.py",
		"./Modules/AvgPool3D.py", "./Modules/AvgPool2D.py", "./Modules/AvgPool1D.py",

		"./Modules/Dropout.py", "./Modules/Dropout2D.py", "./Modules/Cast.py",
		"./Modules/Pad1D.py", "./Modules/Penalty.py", "./Modules/Pad2D.py",

		"./Cost/CTC.py", "./Modules/RNN.py",
		"./Models/Nets/UNet.py", "./Models/Nets/WaveToLetter.py",
		"./Models/Nets/SentiNet.py", "./Models/Nets/Presets/SentiNet.py",

		"./OpenCL/ThirdParty/libclblas.py"
	])

	systemMods = [Utils, MIOpen, CLBlas]
	return exclude, systemMods


def prepareCPUBackend(exclude):
	exclude.discard("./CPU")
	return exclude, []


def prepareIntelBackend(exclude):
	os.environ["OMP_NUM_THREADS"] = str(2)
	from PuzzleLib.Intel.Wrappers import DNNL

	exclude.difference_update(["./CPU", "./Intel"])
	exclude.update(["./Intel/Wrappers/DNNL.py"])

	exclude.update([
		"./Modules/Pad1D.py", "./Modules/Pad2D.py", "./Modules/Embedder.py", "./Modules/PRelu.py", "./Modules/Cast.py",
		"./Modules/Upsample2D.py", "./Modules/Upsample3D.py", "./Modules/MapLRN.py",
		"./Modules/SubtractMean.py", "./Modules/LCN.py", "./Modules/MaxUnpool2D.py", "./Modules/DepthConcat.py",
		"./Modules/BatchNorm.py", "./Modules/BatchNorm1D.py", "./Modules/BatchNorm2D.py", "./Modules/BatchNorm3D.py",
		"./Modules/Sum.py", "./Modules/GroupLinear.py", "./Modules/RNN.py", "./Cost/CTC.py", "./Modules/SpatialTf.py",
		"./Models/Nets/SentiNet.py", "./Models/Nets/Presets/SentiNet.py"
	])

	systemMods = [DNNL]
	return exclude, systemMods


def isValidName(path, filename, exclude):
	if not filename.endswith(".py"):
		return False

	fullname = os.path.join(path, filename)

	if any(file for file in exclude if os.path.commonprefix([fullname, file]) == file):
		return False

	return True


def main():
	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	allowed, exclude, threshold = parseArgs()
	exclude, systemMods = prepareBackend(exclude)

	filenames = []
	exclude = set(os.path.abspath(file) for file in exclude)

	for dirname, subdirs, names in os.walk(os.getcwd()):
		filenames.extend([os.path.join(dirname, file) for file in names if isValidName(dirname, file, exclude)])

	random.shuffle(filenames)
	failure, success, ignored = runUnittests(filenames, systemMods, threshold, allowed=allowed)

	print("Checked %s source files: %s" % (len(filenames), filenames))
	print("Ignored %s files without unittests: %s" % (len(ignored), ignored))

	print("Success on %s files: %s" % (len(success), success))
	print("Failure on %s files: %s" % (len(failure), failure))


if __name__ == "__main__":
	main()
