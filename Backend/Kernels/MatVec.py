from PuzzleLib import Config


addVecToMat = None
argmax = None


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
	memoryPool, matmod = backend.memoryPool, backend.matmod

	def wrapAddVecToMat(vec, mat, axis, out):
		return matmod.addVecToMat(vec, mat, axis, out, memoryPool)

	def wrapArgmax(tensor, axis):
		return matmod.argmax(tensor, axis, memoryPool)

	global addVecToMat, argmax
	addVecToMat = wrapAddVecToMat
	argmax = wrapArgmax


def initCPU():
	import numpy as np
	from PuzzleLib.CPU.CPUArray import CPUArray

	def wrapAddVecToMat(v, m, axis, out):
		if axis == 0:
			v = v[:, np.newaxis]
		elif axis == 1:
			v = v[np.newaxis, :]

		np.add(m.get(copy=False), v.get(copy=False), out=out.get(copy=False))

	def wrapArgmax(mats, axis):
		out = np.empty(mats.shape[:axis] + mats.shape[axis + 1:], dtype=np.int32)
		np.argmax(mats.get(copy=False), axis, out=out)

		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	global addVecToMat, argmax
	addVecToMat = wrapAddVecToMat
	argmax = wrapArgmax


autoinit()
