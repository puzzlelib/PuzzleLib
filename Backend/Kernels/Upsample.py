from PuzzleLib import Config


upsample2d = None
upsample2dBackward = None

upsample3d = None
upsample3dBackward = None


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
	backend = Backend.getBackend(Config.deviceIdx, initmode=2, logger=Config.getLogger())
	memoryPool, upsamplemod = backend.memoryPool, backend.upsamplemod

	def wrapUpsample2d(data, scale, mode):
		return upsamplemod.upsample2d(data, scale, mode, memoryPool)

	def wrapUpsample2dBackward(grad, scale, mode):
		return upsamplemod.upsample2dBackward(grad, scale, mode, memoryPool)

	global upsample2d, upsample2dBackward
	upsample2d = wrapUpsample2d
	upsample2dBackward = wrapUpsample2dBackward

	def wrapUpsample3d(data, scale, mode):
		return upsamplemod.upsample3d(data, scale, mode, memoryPool)

	def wrapUpsample3dBackward(grad, scale, mode):
		return upsamplemod.upsample3dBackward(grad, scale, mode, memoryPool)

	global upsample3d, upsample3dBackward
	upsample3d = wrapUpsample3d
	upsample3dBackward = wrapUpsample3dBackward


def initCPU():
	from PuzzleLib.CPU.Kernels import Upsample2D

	global upsample2d
	upsample2d = Upsample2D.upsample2d


autoinit()
