from PuzzleLib import Config


timeKernel = None


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
	backend = Backend.getBackend(Config.deviceIdx, initmode=0)

	global timeKernel
	timeKernel = backend.timeKernel


def initCPU():
	from PuzzleLib.CPU.Benchmarks import Utils

	global timeKernel
	timeKernel = Utils.timeKernel


autoinit()
