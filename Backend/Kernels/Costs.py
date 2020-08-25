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
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend
	from PuzzleLib.Cuda.Kernels import CTC

	initGPU(Backend, CTC)


def initHip():
	from PuzzleLib.Hip import Backend
	from PuzzleLib.Cuda.Kernels import CTC

	initGPU(Backend, CTC)


def initGPU(Backend, CTC):
	backend = Backend.getBackend(Config.deviceIdx, initmode=2, logger=Config.getLogger())
	memoryPool, costmod, ctcmod = backend.memoryPool, backend.costmod, backend.ctcmod

	global bceKer, hingeKer, smoothL1Ker, l1HingeKer, getAccuracyKernel
	bceKer = backend.bceKer
	hingeKer = backend.hingeKer
	smoothL1Ker = backend.smoothL1Ker
	l1HingeKer = backend.l1HingeKer
	getAccuracyKernel = backend.getAccuracyKernel

	def wrapCrossEntropy(scores, labels, weights, error):
		return costmod.crossEntropy(scores, labels, weights, error, memoryPool)

	def wrapSVM(scores, labels, mode, error):
		return costmod.svm(scores, labels, mode, error, memoryPool)

	global crossEntropyKernel, svmKernel
	crossEntropyKernel = wrapCrossEntropy
	svmKernel = wrapSVM

	def wrapCTC(data, datalen, labels, lengths, blank, error, normalized):
		return ctcmod.ctcLoss(data, datalen, labels, lengths, blank, error, normalized, allocator=memoryPool)

	global ctcLoss, ctcLossTest
	ctcLoss = wrapCTC
	ctcLossTest = CTC.hostCTCLoss


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
