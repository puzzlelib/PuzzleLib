from PuzzleLib import Config


reflectpad1d = None
reflectpad1dBackward = None

reflectpad2d = None
reflectpad2dBackward = None


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
	memoryPool, padmod = backend.memoryPool, backend.padmod

	def wrapReflectPad(data, pad):
		return padmod.reflectpad(data, pad, memoryPool)

	def wrapReflectPadBackward(grad, pad):
		return padmod.reflectpadBackward(grad, pad, memoryPool)

	global reflectpad1d, reflectpad1dBackward, reflectpad2d, reflectpad2dBackward
	reflectpad1d = reflectpad2d = wrapReflectPad
	reflectpad1dBackward = reflectpad2dBackward = wrapReflectPadBackward


def initCPU():
	from PuzzleLib.CPU.Kernels import Pad

	global reflectpad1d
	reflectpad1d = Pad.reflectpad1d

	global reflectpad2d
	reflectpad2d = Pad.reflectpad2d


autoinit()
