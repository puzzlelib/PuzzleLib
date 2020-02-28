from PuzzleLib import Config


GPUArray = None

to_gpu = None
empty = None
zeros = None

minimum = None
maximum = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.GPUArray import GPUArray as CudaArray

	global GPUArray, to_gpu, empty, zeros
	GPUArray = CudaArray
	to_gpu = CudaArray.toGpu
	empty = CudaArray.empty
	zeros = CudaArray.zeros

	global minimum, maximum
	minimum = CudaArray.min
	maximum = CudaArray.max


def initOpenCL():
	from PuzzleLib.OpenCL.Driver import Driver
	from PuzzleLib.OpenCL.Kernels import Templates
	from PuzzleLib.OpenCL.Utils import context, queue

	global GPUArray, to_gpu, empty, zeros
	GPUArray = Driver.Array
	to_gpu = lambda *args, **kwargs: Driver.to_device(queue, *args, **kwargs)
	empty = lambda *args, **kwargs: Driver.empty(queue, *args, **kwargs)
	zeros = lambda *args, **kwargs: Driver.zeros(queue, *args, **kwargs)

	global minimum, maximum
	minimum = lambda ary: Templates.minimum(context, queue, ary)
	maximum = lambda ary: Templates.maximum(context, queue, ary)


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
