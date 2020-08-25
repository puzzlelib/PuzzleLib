import sys, os, traceback, gc, random, importlib
from colorama import Fore, Style

from PuzzleLib import Config
Config.systemLog = True


if "PYCHARM_HOSTED" not in os.environ:
	import colorama
	colorama.init()


def runModuleTest(mod, path, threshold=20):
	itern = 0
	prevpath = os.getcwd()

	try:
		os.chdir(path)

		while True:
			try:
				mod.unittest()

			except Exception as e:
				if isinstance(e, AssertionError):
					traceinfo = traceback.extract_tb(sys.exc_info()[-1])
					_, line, _, text = traceinfo[-1]

					e = "An assert error occurred on line %s in statement:\n'%s'" % (line, text)

				print(
					Fore.YELLOW + "%s unittest failed on try #%s with reason: %s" % (mod, itern + 1, e) +
					Style.RESET_ALL
				)

				if itern < threshold - 1:
					itern += 1
					continue

				else:
					print(Fore.RED + "... threshold limit exceeded, skipping ..." + Style.RESET_ALL)
					return False

			print(Fore.LIGHTGREEN_EX + "%s unittest finished successfully" % mod + Style.RESET_ALL)
			return True

	finally:
		os.chdir(prevpath)


def runUnittests(filenames, basepath, threshold=20):
	from PuzzleLib.Backend.gpuarray import setupDebugAllocator

	print("Setting debug allocator ...")
	setupDebugAllocator()

	gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

	failure, success, ignored = [], [], []
	for filename in filenames:
		importname = os.path.splitext(os.path.relpath(filename, basepath))[0].replace(os.sep, ".")

		try:
			mod = importlib.import_module("%s.%s" % (Config.libname, importname))

			if hasattr(mod, "unittest") and not hasattr(mod, "main"):
				print("%s has unittest. Starting ..." % filename)

				dstContainer = success if runModuleTest(mod, os.path.dirname(filename), threshold) else failure
				dstContainer.append(filename)

				gc.collect()

			else:
				print("Skipping %s ..." % filename)
				ignored.append(filename)

		except Exception as e:
			traceinfo = traceback.extract_tb(sys.exc_info()[-1])
			_, line, _, _ = traceinfo[-1]

			e = "(line: %s, type: %s): %s" % (line, type(e), e)
			print(Fore.RED + "%s testing failed with reason: %s" % (importname, e) + Style.RESET_ALL)

			failure.append(filename)

	return failure, success, ignored


def prepareBackend(exclude):
	if Config.backend == Config.Backend.cuda:
		exclude = prepareCudaBackend(exclude)

	elif Config.backend == Config.Backend.hip:
		exclude = prepareHipBackend(exclude)

	elif Config.backend == Config.Backend.cpu:
		exclude = prepareCPUBackend(exclude)

	elif Config.backend == Config.Backend.intel:
		exclude = prepareIntelBackend(exclude)

	else:
		raise NotImplementedError(Config.backend)

	return exclude


def prepareCudaBackend(exclude):
	exclude.discard("Cuda")
	return exclude


def prepareHipBackend(exclude):
	exclude.discard("Hip")
	exclude.update([
		"Modules/Pad2D.py", "Modules/Split.py", "Modules/Slice.py",
		"Modules/MaxPool3D.py", "Modules/AvgPool3D.py", "Modules/SpatialTf.py",
		"Modules/CrossMapLRN.py", "Modules/LCN.py"
	])

	return exclude


def prepareCPUBackend(exclude):
	exclude.discard("CPU")

	exclude.update([
		"Modules/Conv1D.py", "Modules/Conv2D.py", "Modules/Conv3D.py",
		"Modules/Deconv1D.py", "Modules/Deconv2D.py", "Modules/Deconv3D.py",
		"Modules/BatchNorm1D.py", "Modules/BatchNorm2D.py", "Modules/BatchNorm3D.py", "Modules/BatchNorm.py",
		"Modules/MaxPool1D.py", "Modules/MaxPool2D.py", "Modules/MaxPool3D.py",
		"Modules/AvgPool1D.py", "Modules/AvgPool2D.py", "Modules/AvgPool3D.py",
		"Modules/Pad1D.py", "Modules/Pad2D.py", "Modules/Upsample2D.py", "Modules/Upsample3D.py",
		"Modules/InstanceNorm2D.py", "Modules/MaxUnpool2D.py", "Modules/Cast.py", "Modules/Sum.py",
		"Modules/DepthConcat.py", "Modules/LCN.py","Modules/PRelu.py", "Modules/RNN.py", "Modules/Embedder.py",
		"Modules/SpatialTf.py", "Modules/Gelu.py", "Modules/CrossMapLRN.py", "Modules/SoftMax.py", "Modules/MapLRN.py",
		"Modules/GroupLinear.py", "Modules/SubtractMean.py",

		"Models/Nets/NiN.py", "Models/Nets/VGG.py", "Models/Nets/ResNet.py", "Models/Nets/Inception.py",
		"Models/Nets/WaveToLetter.py", "Models/Nets/MiniYolo.py", "Models/Nets/UNet.py",
		"Models/Nets/SentiNet.py", "Models/Nets/Presets/SentiNet.py", "Models/Misc/RBM.py",

		"Cost/CrossEntropy.py", "Cost/BCE.py", "Cost/L1Hinge.py", "Cost/SmoothL1.py", "Cost/KLDivergence.py",
		"Cost/Hinge.py", "Cost/SVM.py", "Cost/CTC.py",

		"Containers/Sequential.py", "Containers/Parallel.py", "Containers/Graph.py",
		"Handlers/Trainer.py", "Handlers/Validator.py", "Passes/ConvertToGraph.py",

		"Optimizers/AdaDelta.py", "Optimizers/AdaGrad.py", "Optimizers/Adam.py",
		"Optimizers/SGD.py", "Optimizers/MomentumSGD.py", "Optimizers/NesterovSGD.py",
		"Optimizers/RMSProp.py", "Optimizers/RMSPropGraves.py", "Optimizers/SMORMS3.py"
	])

	return exclude


def prepareIntelBackend(exclude):
	os.environ["OMP_NUM_THREADS"] = str(2)
	exclude.difference_update(["CPU", "Intel"])

	exclude.update([
		"Modules/Pad1D.py", "Modules/Pad2D.py", "Modules/Embedder.py", "Modules/PRelu.py", "Modules/Cast.py",
		"Modules/Upsample2D.py", "Modules/Upsample3D.py", "Modules/MapLRN.py", "Modules/Gelu.py",
		"Modules/SubtractMean.py", "Modules/LCN.py", "Modules/MaxUnpool2D.py", "Modules/DepthConcat.py",
		"Modules/BatchNorm.py", "Modules/BatchNorm1D.py", "Modules/BatchNorm2D.py", "Modules/BatchNorm3D.py",
		"Modules/Sum.py", "Modules/GroupLinear.py", "Modules/RNN.py", "Modules/SpatialTf.py",
		"Cost/CTC.py",
		"Models/Nets/SentiNet.py", "Models/Nets/Presets/SentiNet.py"
	])

	return exclude


def gatherTestableFiles(exclude):
	filenames = []
	exclude = set(os.path.abspath(file) for file in exclude)

	for dirname, subdirs, names in os.walk(os.getcwd()):
		files = (os.path.join(dirname, file) for file in names)
		filenames.extend(file for file in files if isTestableFile(file, exclude))

	random.shuffle(filenames)
	return filenames


def isTestableFile(filename, exclude):
	if not filename.endswith(".py") or any(file for file in exclude if os.path.commonprefix((filename, file)) == file):
		return False

	return True


def main():
	basepath = os.path.dirname(os.path.abspath(__file__))
	os.chdir(basepath)

	exclude = {
		"Converter",
		"Cuda", "Hip", "CPU", "Intel"
	}

	filenames = gatherTestableFiles(prepareBackend(exclude))
	failure, success, ignored = runUnittests(filenames, basepath, threshold=20)

	print("Checked %s source files: %s" % (len(filenames), filenames))
	print("Ignored %s files without unittests: %s" % (len(ignored), ignored))

	print("Success on %s files: %s" % (len(success), success))
	print("Failure on %s files: %s" % (len(failure), failure))


if __name__ == "__main__":
	main()
