from PuzzleLib import Config


instanceNorm2d = None
instanceNorm2dBackward = None


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
	initGPU(Backend)


def initHip():
	from PuzzleLib.Hip import Backend
	initGPU(Backend)


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=2)
	memoryPool = backend.memoryPool

	def wrapInstanceNorm2d(data, scale, bias, epsilon=1e-5):
		return backend.instanceNorm2d(data, scale.ravel(), bias.ravel(), epsilon, allocator=memoryPool)

	def wrapInstanceNorm2dBackward(grad, data, extscale, savemean, saveinvvar, epsilon, affine):
		return backend.instanceNorm2dBackward(
			grad, data, extscale, savemean, saveinvvar, epsilon, affine, allocator=memoryPool
		)

	global instanceNorm2d, instanceNorm2dBackward
	instanceNorm2d = wrapInstanceNorm2d
	instanceNorm2dBackward = wrapInstanceNorm2dBackward


def initCPU():
	pass


def initIntel():
	from PuzzleLib.Intel.Wrappers import DNNLInstanceNorm

	def wrapInstanceNorm2d(data, scale, bias, epsilon):
		result = DNNLInstanceNorm.instanceNorm2d(data, scale, bias, epsilon)

		outdata, savemean, savevar, extscale, extbias, desc = result
		return outdata, savemean, savevar, (extscale, extbias, desc)

	def wrapInstanceNorm2dBackward(grad, data, exts, savemean, savevar, epsilon, affine):
		extscale, extbias, desc = exts
		return DNNLInstanceNorm.instanceNorm2dBackward(
			grad, data, extscale, extbias, savemean, savevar, epsilon, desc, affine
		)

	global instanceNorm2d, instanceNorm2dBackward
	instanceNorm2d = wrapInstanceNorm2d
	instanceNorm2dBackward = wrapInstanceNorm2dBackward


autoinit()
