from PuzzleLib import Config


prelu = None
preluBackwardData = None
preluBackwardParams = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend
	initGPU(Backend)


def initHip():
	from PuzzleLib.Hip import Backend
	initGPU(Backend)


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=2)
	memoryPool, prelumod = backend.memoryPool, backend.prelumod

	def wrapPRelu(data, slopes, inplace, sharedMaps):
		return prelumod.prelu(data, slopes, inplace, sharedMaps, memoryPool)

	def wrapPReluBackwardData(grad, slopes, indata, sharedMaps):
		return prelumod.preluBackwardData(grad, slopes, indata, sharedMaps, memoryPool)

	def wrapPReluBackwardParams(indata, outgrad, sharedMaps):
		return prelumod.preluBackwardParams(indata, outgrad, sharedMaps, memoryPool)

	global prelu, preluBackwardData, preluBackwardParams
	prelu = wrapPRelu
	preluBackwardData = wrapPReluBackwardData
	preluBackwardParams = wrapPReluBackwardParams


def initCPU():
	pass


autoinit()
