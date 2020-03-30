from multiprocessing import Process, SimpleQueue


def runGrid(target, size, *args, devices=None, **kwargs):
	gridinfo = generateGridInfo(size, devices)
	nodes = [Process(target=nodeRunner, args=(target, nodeinfo) + args, kwargs=kwargs) for nodeinfo in gridinfo]

	for node in nodes:
		node.start()

	for node in nodes:
		node.join()


def generateGridInfo(size, devices):
	devices = range(size) if devices is None else devices

	queues = [(SimpleQueue(), SimpleQueue()) for _ in range(size - 1)]
	parent = ParentNode(0, size, devices[0], queues)

	nodes = [ChildNode(index + 1, size, devices[index + 1], queues[index]) for index in range(size - 1)]
	return [parent] + nodes


def nodeRunner(target, nodeinfo, *args, **kwargs):
	from PuzzleLib import Config

	Config.allowMultiContext = True
	Config.deviceIdx = nodeinfo.device

	try:
		target(nodeinfo, *args, **kwargs)

	finally:
		nodeinfo.close()


class NodeInfo:
	def __init__(self, index, gridsize, device, queues):
		self.index = index
		self.gridsize = gridsize

		self.device = device
		self.queues = queues

		self.outTensors, self.inTensors = {}, {}


	def close(self):
		for mapped, _ in self.inTensors.values():
			mapped.free()


	def meanValue(self, value):
		raise NotImplementedError()


	def broadcastBuffer(self, name, buffer):
		raise NotImplementedError()


	def sumTensor(self, name, tensor):
		raise NotImplementedError()


	def recvBuffer(self, name, queue, buffer=None):
		from PuzzleLib.Backend.Utils import backend

		parentname, bufipc, bufsize, args = queue.get()
		assert name == parentname

		cache = self.inTensors.get(name, None)

		if cache is None:
			cache = (backend.Driver.allocateFromIPCHandle(bufipc, bufsize), None)
			self.inTensors[name] = cache

		mapped, _ = cache

		if buffer is not None:
			mapped.copy(dst=buffer)
			backend.Driver.Device.synchronize()

			mapped = buffer

		return mapped, args


	def sendBuffer(self, name, buffer, queue, *args):
		from PuzzleLib.Backend.Utils import backend

		if name not in self.outTensors:
			self.outTensors[name] = buffer
			bufipc = buffer.getIPCHandle()
		else:
			assert self.outTensors[name] is buffer
			bufipc = None

		backend.Driver.Device.synchronize()
		queue.put((name, bufipc, buffer.size, args))


class ParentNode(NodeInfo):
	def meanValue(self, value):
		value += sum(ctopQueue.get() for _, ctopQueue in self.queues)
		value /= self.gridsize

		for ptocQueue, _ in self.queues:
			ptocQueue.put(value)

		return value


	def broadcastBuffer(self, name, buffer):
		for ptocQueue, _ in self.queues:
			self.sendBuffer(name, buffer, ptocQueue)

		for _, ctopQueue in self.queues:
			childname = ctopQueue.get()
			assert name == childname


	def sumTensor(self, name, tensor):
		from PuzzleLib.Backend.Blas import addVectorToVector
		from PuzzleLib.Backend.gpuarray import GPUArray

		beta = 1.0 / self.gridsize

		for index, (_, ctopQueue) in enumerate(self.queues):
			buffer, (shape, dtype) = self.recvBuffer(name, ctopQueue)
			assert shape == tensor.shape and dtype == tensor.dtype

			childTensor = GPUArray(shape, dtype, gpudata=buffer)
			addVectorToVector(tensor, childTensor, out=tensor, alpha=beta if index == 0 else 1.0, beta=beta)

		self.broadcastBuffer(name, tensor.gpudata)


class ChildNode(NodeInfo):
	def meanValue(self, value):
		ptocQueue, ctopQueue = self.queues

		ctopQueue.put(value)
		return ptocQueue.get()


	def broadcastBuffer(self, name, buffer):
		ptocQueue, ctopQueue = self.queues

		self.recvBuffer(name, ptocQueue, buffer=buffer)
		ctopQueue.put(name)


	def sumTensor(self, name, tensor):
		_, ctopQueue = self.queues

		self.sendBuffer(name, tensor.gpudata, ctopQueue, tensor.shape, tensor.dtype)
		self.broadcastBuffer(name, tensor.gpudata)
