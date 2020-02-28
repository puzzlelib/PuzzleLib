from PuzzleLib import Config


bceKer = None
hingeKer = None
smoothL1Ker = None
l1HingeKer = None

getAccuracyKernel = None
crossEntropyKernel = None
svmKernel = None

ctcLoss = None
ctcLossTest = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.Kernels import Costs, CTC

	global bceKer, hingeKer, smoothL1Ker, l1HingeKer, getAccuracyKernel, crossEntropyKernel, svmKernel
	bceKer = Costs.bceKer
	hingeKer = Costs.hingeKer
	smoothL1Ker = Costs.smoothL1Ker
	l1HingeKer = Costs.l1HingeKer
	getAccuracyKernel = Costs.getAccuracyKernel
	crossEntropyKernel = Costs.crossEntropy
	svmKernel = Costs.svm

	global ctcLoss, ctcLossTest
	ctcLoss = CTC.ctcLoss
	ctcLossTest = CTC.ctcLossTest


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import Costs

	global bceKer, hingeKer, smoothL1Ker, l1HingeKer, getAccuracyKernel, crossEntropyKernel, svmKernel
	bceKer = Costs.bceKer
	hingeKer = Costs.hingeKer
	smoothL1Ker = Costs.smoothL1Ker
	l1HingeKer = Costs.l1HingeKer
	getAccuracyKernel = Costs.getAccuracyKernel
	crossEntropyKernel = Costs.crossEntropy
	svmKernel = Costs.svm


def initCPU():
	pass


def initIntel():
	from PuzzleLib.Intel.Kernels import Costs

	global bceKer, hingeKer, smoothL1Ker, l1HingeKer, getAccuracyKernel, crossEntropyKernel, svmKernel
	bceKer = Costs.bceKer
	hingeKer = Costs.hingeKer
	smoothL1Ker = Costs.smoothL1Ker
	l1HingeKer = Costs.l1HingeKer
	getAccuracyKernel = Costs.getAccuracyKernel
	crossEntropyKernel = Costs.crossEntropy
	svmKernel = Costs.svm


autoinit()
