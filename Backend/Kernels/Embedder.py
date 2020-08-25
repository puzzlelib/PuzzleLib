from PuzzleLib import Config


embed = None
embedBackwardParams = None


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
	memoryPool, embedmod = backend.memoryPool, backend.embedmod

	def wrapEmbed(data, W):
		return embedmod.embed(data, W, memoryPool)

	global embed, embedBackwardParams
	embed = wrapEmbed
	embedBackwardParams = embedmod.embedBackwardParams


def initCPU():
	pass


autoinit()
