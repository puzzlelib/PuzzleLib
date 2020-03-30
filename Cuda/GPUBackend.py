import sys, time
import numpy as np

from PuzzleLib import Config
from PuzzleLib.Cuda.Utils import prod, QueueManager

from PuzzleLib.Cuda.Kernels import ElementWise
from PuzzleLib.Cuda.Kernels.Costs import bce, hinge, smoothL1, l1Hinge, CostModule
from PuzzleLib.Cuda.Kernels.CTC import CTCModule
from PuzzleLib.Cuda.Kernels.Embedder import EmbedModule
from PuzzleLib.Cuda.Kernels.MatVec import MatModule
from PuzzleLib.Cuda.Kernels.Pad import PadModule
from PuzzleLib.Cuda.Kernels.Pool import PoolModule
from PuzzleLib.Cuda.Kernels.PRelu import PReluModule
from PuzzleLib.Cuda.Kernels.Upsample import UpsampleModule


class GPUBackend:
	BackendName = None

	warpSize = None
	nthreads = None

	Driver = None
	GPUArray = None
	Error = None

	SourceModule = None
	ElementwiseKernel = None
	ElementHalf2Kernel = None
	ReductionKernel = None

	SharedArray = None
	Rand, Blas, Dnn = None, None, None

	ConvPerf = None
	ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo = None, None, None

	RNNAlgo, RNNMode, DirectionMode = None, None, None


	def __init__(self, deviceIdx, initmode=0):
		self.deviceIdx = deviceIdx

		ndevices = self.Driver.Device.count()
		if ndevices == 0:
			raise self.Error("No %s enabled device found" % self.BackendName)

		if deviceIdx >= ndevices:
			raise self.Error("Invalid %s config device index" % self.BackendName)

		self.device = self.Driver.Device(deviceIdx).set()
		print("[%s] Using device #%s (%s)" % (Config.libname, deviceIdx, self.device.name()), flush=True)

		if Config.systemLog:
			print(
				"[%s] Created %s context (Using driver version: %s)" %
				(Config.libname, self.BackendName, self.Driver.getDriverVersion()), flush=True
			)

		self.memoryPool = self.Driver.MemoryPool()

		rngtype, seed = self.Rand.RAND_RNG_PSEUDO_XORWOW, int(np.random.randint(sys.maxsize, dtype=np.intp))
		self.globalRng = self.Rand.RandomNumberGenerator(type=rngtype, seed=seed)

		if Config.systemLog:
			print(
				"[%s] Created %s global rng (type=%s, seed=%s)" %
				(Config.libname, self.Rand.__name__, self.globalRng.type, hex(self.globalRng.seed)), flush=True
			)

		self.streamManager = QueueManager(objtype=self.Driver.Stream)
		self.eventManager = QueueManager(objtype=self.Driver.Event)

		self.blas, self.dnn = None, None

		self.costmod = None
		self.ctcmod = None
		self.embedmod = None
		self.matmod = None
		self.padmod = None
		self.poolmod = None
		self.prelumod = None
		self.upsamplemod = None

		self.bceKer = None
		self.hingeKer = None
		self.smoothL1Ker = None
		self.l1HingeKer = None
		self.getAccuracyKernel = None

		self.sigmoidKer = None
		self.sigmoidDerKer = None
		self.tanhKer = None
		self.tanhDerKer = None
		self.reluKer = None
		self.reluDerKer = None
		self.leakyReluKer = None
		self.leakyReluDerKer = None
		self.eluKer = None
		self.eluDerKer = None
		self.softPlusKer = None
		self.softPlusDerKer = None
		self.clipKer = None
		self.clipDerKer = None
		self.geluKer = None
		self.geluDerKer = None

		self.dropoutKer = None
		self.dropout2dKer = None
		self.rbmKer = None
		self.absKer = None
		self.toVectorAddVectorKer = None

		self.classicMomSGDKer = None
		self.nesterovMomSGDKer = None
		self.rmspropKer = None
		self.adamKer = None
		self.rmspropGravesKer = None
		self.adagradKer = None
		self.adadeltaKer = None
		self.smorms3Ker = None

		self.weightDecayKer = None
		self.linearKer = None
		self.addKer = None
		self.mulKer = None
		self.l1penaltyKer = None
		self.l1gradKer = None

		self.castFP16toFP32 = None
		self.castFP32toFP16 = None

		self.initmode = 0
		self.updateBackend(initmode)


	def updateBackend(self, initmode):
		if initmode > 0 >= self.initmode:
			self.initLibs()

		if initmode > 1 >= self.initmode:
			self.initKernels()

		self.initmode = max(initmode, self.initmode)


	def initLibs(self):
		self.blas = self.Blas.BlasContext().enableTensorOps(True)
		if Config.systemLog:
			print(
				"[%s] Created %s context (Using version: %s)" %
				(Config.libname, self.Blas.__name__, self.blas.getVersion())
			)

		self.dnn = self.Dnn.DnnContext().enableTensorOps(True)
		if Config.systemLog:
			print(
				"[%s] Created %s context (Using version: %s)" %
				(Config.libname, self.Dnn.__name__, self.dnn.getVersion())
			)


	def initKernels(self):
		self.costmod = CostModule(self)
		self.ctcmod = CTCModule(self)
		self.embedmod = EmbedModule(self)
		self.matmod = MatModule(self)
		self.padmod = PadModule(self)
		self.poolmod = PoolModule(self)
		self.prelumod = PReluModule(self.matmod)
		self.upsamplemod = UpsampleModule(self)

		self.bceKer = bce(self.ElementwiseKernel)
		self.hingeKer = hinge(self.ElementwiseKernel)
		self.smoothL1Ker = smoothL1(self.ElementwiseKernel)
		self.l1HingeKer = l1Hinge(self.ElementwiseKernel)
		self.getAccuracyKernel = self.costmod.getAccuracyKernel

		self.sigmoidKer = ElementWise.sigmoid(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.sigmoidDerKer = ElementWise.sigmoidDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.tanhKer = ElementWise.tanh(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.tanhDerKer = ElementWise.tanhDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.reluKer = ElementWise.relu(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.reluDerKer = ElementWise.reluDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.leakyReluKer = ElementWise.leakyRelu(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.leakyReluDerKer = ElementWise.leakyReluDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.eluKer = ElementWise.elu(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.eluDerKer = ElementWise.eluDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.softPlusKer = ElementWise.softPlus(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.softPlusDerKer = ElementWise.softPlusDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.clipKer = ElementWise.clip(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.clipDerKer = ElementWise.clipDer(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.geluKer = ElementWise.gelu(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.geluDerKer = ElementWise.geluDer(self.ElementwiseKernel, self.ElementHalf2Kernel)

		self.dropoutKer = ElementWise.dropout(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.dropout2dKer = ElementWise.dropout2d(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.rbmKer = ElementWise.rbmKer(self.ElementwiseKernel)
		self.absKer = ElementWise.absKer(self.ElementwiseKernel)
		self.toVectorAddVectorKer = ElementWise.toVectorAddVector(self.ElementwiseKernel, self.ElementHalf2Kernel)

		self.classicMomSGDKer = ElementWise.classicMomSGD(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.nesterovMomSGDKer = ElementWise.nesterovMomSGD(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.rmspropKer = ElementWise.rmsprop(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.adamKer = ElementWise.adam(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.rmspropGravesKer = ElementWise.rmspropGraves(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.adagradKer = ElementWise.adagrad(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.adadeltaKer = ElementWise.adadelta(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.smorms3Ker = ElementWise.smorms3(self.ElementwiseKernel, self.ElementHalf2Kernel)

		self.weightDecayKer = ElementWise.weightDecayKer(self.ElementwiseKernel)
		self.linearKer = ElementWise.linear(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.addKer = ElementWise.add(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.mulKer = ElementWise.mul(self.ElementwiseKernel, self.ElementHalf2Kernel)
		self.l1penaltyKer = ElementWise.l1penaltyKer(self.ElementwiseKernel)
		self.l1gradKer = ElementWise.l1gradKer(self.ElementwiseKernel)

		self.castFP16toFP32 = ElementWise.castFP16toFP32(self.ElementwiseKernel)
		self.castFP32toFP16 = ElementWise.castFP32toFP16(self.ElementwiseKernel)


	@staticmethod
	def dtypesSupported():
		return [(np.float32, 1e-5), (np.float16, 1e-2)]


	@staticmethod
	def copy(dest, source, allocator=None):
		if dest is None:
			return source.copy(allocator=allocator)
		else:
			dest.set(source)
			return dest


	def getDeviceComputeCap(self, index):
		return self.Driver.Device(index).computeCapability()


	def fillUniform(self, data, minval=0.0, maxval=1.0, rng=None):
		assert data.dtype == np.float32

		rng = self.globalRng if rng is None else rng
		rng.fillUniform(data)

		dtype = data.dtype
		self.linearKer(dtype)(data, data, dtype.type(maxval - minval), dtype.type(minval))


	def fillNormal(self, data, mean=0.0, stddev=1.0, rng=None):
		rng = self.globalRng if rng is None else rng
		rng.fillNormal(data, mean=mean, stddev=stddev)


	def dstack(self, tup, allocator=None):
		return self.concatenate(tup, axis=2, allocator=allocator)


	def hstack(self, tup, allocator=None):
		return self.concatenate(tup, axis=1, allocator=allocator)


	def vstack(self, tup, allocator=None):
		return self.concatenate(tup, axis=0, allocator=allocator)


	def dsplit(self, ary, sections, allocator=None):
		return self.split(ary, sections, axis=2, allocator=allocator)


	def hsplit(self, ary, sections, allocator=None):
		return self.split(ary, sections, axis=1, allocator=allocator)


	def vsplit(self, ary, sections, allocator=None):
		return self.split(ary, sections, axis=0, allocator=allocator)


	def concatenate(self, tup, axis, out=None, allocator=None):
		ary = tup[0]

		dtype, reducedShape = ary.dtype, ary.shape
		reducedShape = reducedShape[:axis] + reducedShape[axis + 1:]

		assert all(a.dtype == dtype and a.shape[:axis] + a.shape[axis + 1:] == reducedShape for a in tup[1:])

		concatDim = sum(a.dimAt(axis) for a in tup)
		shape = reducedShape[:axis] + (concatDim, ) + reducedShape[axis:]

		if out is None:
			out = self.GPUArray.empty(shape, dtype=dtype, allocator=allocator)
		else:
			assert out.shape == shape and out.dtype == dtype

		dstPitch = out.strideAt(axis - 1) if axis > 0 else out.nbytes
		height = prod(shape[:axis])

		stride = 0

		for a in tup:
			srcPitch = width = a.strideAt(axis - 1) if axis > 0 else a.nbytes

			self.Driver.memcpy2D(width, height, a.gpudata, srcPitch, out.gpudata, dstPitch, dstX=stride)
			stride += width

		return out


	def split(self, ary, sections, axis, allocator=None):
		shape = ary.shape
		assert sum(sections) == shape[axis]

		outs = [
			self.GPUArray.empty(shape[:axis] + (sec, ) + shape[axis + 1:], dtype=ary.dtype, allocator=allocator)
			for sec in sections
		]

		srcPitch = ary.strideAt(axis - 1) if axis > 0 else ary.nbytes
		height = prod(shape[:axis])

		stride = 0

		for out in outs:
			dstPitch = width = out.strideAt(axis - 1) if axis > 0 else out.nbytes

			self.Driver.memcpy2D(width, height, ary.gpudata, srcPitch, out.gpudata, dstPitch, srcX=stride)
			stride += width

		return outs


	def tile(self, ary, repeats, axis, allocator=None):
		return self.concatenate([ary] * repeats, axis=axis, allocator=allocator)


	def timeKernel(self, func, args, kwargs=None, looplength=1000, log=True, logname=None, normalize=False,
				   hotpass=True):
		kwargs = {} if kwargs is None else kwargs

		if hotpass:
			func(*args, **kwargs)

		start, end = self.Driver.Event(), self.Driver.Event()

		hostStart = time.time()
		start.record()

		for _ in range(looplength):
			func(*args, **kwargs)

		end.record()
		hostEnd = time.time()

		end.synchronize()
		millisInSec = 1e-3

		devsecs = start.timeTill(end) * millisInSec
		hostsecs = hostEnd - hostStart

		if logname is None:
			funcname = func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
			logname = "%s.%s" % (func.__module__, funcname)

		if normalize:
			devsecs /= looplength
			hostsecs /= looplength

		if log:
			print("%s device time: %s secs" % (logname, devsecs))
			print("%s host time: %s secs" % (logname, hostsecs))

		return devsecs, hostsecs


	def convNdbenchmark(self, datashape, Wshape, dtype, stride=1, pad=0, dilation=1, groups=1, algoCount=10):
		results = self.dnn.convNdbenchmark(datashape, Wshape, dtype, stride, pad, dilation, groups, algoCount)
		results = tuple(
			[self.ConvPerf(algotype(values[0]), *values[1:]) for values in subresults] for algotype, subresults in
			zip((self.ConvFwdAlgo, self.ConvBwdDataAlgo, self.ConvBwdFilterAlgo), results)
		)

		return results


	def instanceNorm2d(self, data, scale, bias, epsilon=1e-5, out=None, allocator=None):
		batchsize, maps, height, width = data.shape
		extmaps = batchsize * maps

		indata = data.reshape(1, extmaps, height, width)

		mean = self.GPUArray.empty((extmaps, ), dtype=np.float32, allocator=allocator)
		var = self.GPUArray.empty((extmaps, ), dtype=np.float32, allocator=allocator)

		if batchsize > 1:
			scale = self.tile(scale, batchsize, axis=0, allocator=allocator)
			bias = self.tile(bias, batchsize, axis=0, allocator=allocator)

		outdata, savemean, saveinvvar = self.dnn.batchNormNd(
			indata, mean, var, scale, bias, epsilon, test=False, out=out, allocator=allocator
		)
		return outdata.reshape(data.shape), savemean, saveinvvar, scale


	def instanceNorm2dBackward(self, grad, data, extscale, savemean, saveinvvar, epsilon, affine=True,
							   out=None, allocator=None):
		batchsize, maps, height, width = grad.shape
		extmaps = batchsize * maps

		outgrad = grad.reshape(1, extmaps, height, width)
		indata = data.reshape(1, extmaps, height, width)

		ingrad, scalegrad, bgrad = self.dnn.batchNormNdBackward(
			outgrad, indata, extscale, savemean, saveinvvar, epsilon, out=out, allocator=allocator
		)

		if affine and batchsize > 1:
			scalegrad = self.matmod.matsum(scalegrad.reshape(batchsize, -1), axis=0, allocator=allocator)
			bgrad = self.matmod.matsum(bgrad.reshape(batchsize, -1), axis=0, allocator=allocator)

		return (ingrad.reshape(grad.shape), scalegrad, bgrad) if affine else ingrad.reshape(grad.shape)


	def createRnn(self, insize, hsize, dtype, layers=1, algo=None, mode=None, direction=None, dropout=0.0,
				  seed=0, batchsize=0):
		raise NotImplementedError()


	def acquireRnnParams(self, rnn, W):
		raise NotImplementedError()


	def updateRnnParams(self, rnn, W, params):
		pass


	def deviceSupportsBatchHint(self):
		return False
