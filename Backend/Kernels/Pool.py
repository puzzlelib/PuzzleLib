from PuzzleLib import Config


maxpool2d = None
maxpool2dBackward = None
maxunpool2d = None
maxunpool2dBackward = None


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
	memoryPool, poolmod = backend.memoryPool, backend.poolmod

	def wrapMaxPool2d(data, size, stride, pad):
		return poolmod.maxpool2d(data, size, stride, pad, memoryPool)

	def wrapMaxPool2dBackward(grad, origshape, mask, size, stride, pad):
		return poolmod.maxpool2dBackward(grad, origshape, mask, size, stride, pad, memoryPool)

	global maxpool2d, maxpool2dBackward
	maxpool2d = wrapMaxPool2d
	maxpool2dBackward = wrapMaxPool2dBackward

	def wrapMaxUnpool2d(data, origshape, mask):
		return poolmod.maxunpool2d(data, origshape, mask, memoryPool)

	def wrapMaxUnpool2dBackward(grad, poolshape, mask):
		return poolmod.maxunpool2dBackward(grad, poolshape, mask, memoryPool)

	global maxunpool2d, maxunpool2dBackward
	maxunpool2d = wrapMaxUnpool2d
	maxunpool2dBackward = wrapMaxUnpool2dBackward


def initCPU():
	pass


autoinit()
