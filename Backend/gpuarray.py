from PuzzleLib import Config


GPUArray = None

to_gpu = None
empty = None
zeros = None

minimum = None
maximum = None


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

	global GPUArray, to_gpu, empty, zeros
	GPUArray = backend.GPUArray
	to_gpu = backend.GPUArray.toGpu
	empty = backend.GPUArray.empty
	zeros = backend.GPUArray.zeros

	global minimum, maximum
	minimum = backend.GPUArray.min
	maximum = backend.GPUArray.max


def initCPU():
	from PuzzleLib.CPU.CPUArray import CPUArray

	global GPUArray, to_gpu, empty, zeros
	GPUArray = CPUArray
	to_gpu = CPUArray.toDevice
	empty = CPUArray.empty
	zeros = CPUArray.zeros

	global minimum, maximum
	minimum = CPUArray.minimum
	maximum = CPUArray.maximum


autoinit()
