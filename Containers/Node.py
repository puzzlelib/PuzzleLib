from PuzzleLib.Backend import gpuarray, Blas


class NodeError(Exception):
	pass


class Node:
	def __init__(self, mod, parents=None, name=None):
		self.module = mod
		self.rename = name

		self.fwdVisited = False
		self.bwdVisited = False

		self.data = None
		self.grad = None

		self.fwds = []
		self.bwds = []

		self.addBackwards(parents)


	def addBackwards(self, nodes):
		if nodes is None:
			return

		if isinstance(nodes, Node):
			nodes.addForward((self, None))
			self.bwds.append((nodes, None))

		elif isinstance(nodes, tuple):
			node, slots = nodes
			if not isinstance(slots, (list, type(None))):
				slots = [slots]

			node.addForward((self, slots))
			self.bwds.append((node, slots))

		elif isinstance(nodes, list):
			for node in nodes:
				self.addBackwards(node)

		else:
			raise NodeError("Unrecognized parent object type %s" % type(nodes).__name__)


	def addForward(self, node):
		self.fwds.append(node)


	@property
	def name(self):
		return self.module.name if self.rename is None else self.rename


	def forward(self, data):
		self.traverseForward(self, Node.updateData, data)


	def updateData(self, data):
		if len(self.bwds) > 0:
			if len(self.bwds) == 1 and self.bwds[0][1] is None:
				data = self.bwds[0][0].data
			else:
				data = []
				for node, slots in self.bwds:
					data.extend([node.data] if slots is None else (node.data[slot] for slot in slots))

		self.data = self.module(data)


	def dataShapeFrom(self, inshapes, shapes, onmodule):
		if len(self.bwds) == 0:
			shape = inshapes[self.name]
		else:
			shape = []
			for node, slots in self.bwds:
				shape.extend([shapes[node.name]] if slots is None else (shapes[node.name][slot] for slot in slots))

			if len(self.bwds) == 1:
				shape = shape[0]

		outshape = self.module.dataShapeFrom(shape)
		if onmodule is not None:
			onmodule(self.module, shape)

		shapes[self.name] = outshape


	def backward(self, grad=None, updParamGrads=True, updGrad=True, scale=1.0, momentum=0.0):
		self.traverseBackward(self, Node.updateGrad, grad, updParamGrads, updGrad, scale, momentum)


	def updateGrad(self, grad, updParamGrads, updGrad, scale, momentum):
		grad = self.buildOutGrad(grad)

		updGrad = updGrad if len(self.bwds) == 0 else True
		self.module.backward(grad, updParamGrads=updParamGrads, updGrad=updGrad, scale=scale, momentum=momentum)

		self.grad = self.routeInGrad(self.module.grad)


	def buildOutGrad(self, grad):
		if len(self.fwds) == 0:
			return grad

		grad = [[] for _ in range(len(self.data) if isinstance(self.data, list) else 1)]

		for node, slots in self.fwds:
			if slots is not None:
				for slot in slots:
					grad[slot].append(node.grad[self.name][slot])

			else:
				for i, gr in enumerate(node.grad[self.name]):
					grad[i].append(gr)

		for i, grads in enumerate(grad):
			if len(grads) > 1:
				gr = gpuarray.copy(None, grads[0])

				for j in range(1, len(grads)):
					Blas.toVectorAddVector(gr.ravel(), grads[j].ravel())

			else:
				gr = grads[0]

			grad[i] = gr

		if len(grad) == 1:
			grad = grad[0]

		return grad


	def routeInGrad(self, grad):
		if len(self.bwds) == 0:
			return grad

		grad = grad if isinstance(grad, list) else [grad]
		routedgrad = {}

		i = 0
		for node, slots in self.bwds:
			if slots is None:
				ln = len(node.data) if isinstance(node.data, list) else 1

				routedgrad[node.name] = grad[i:i + ln]
				i += ln

			else:
				d = {slot: grad[i + j] for j, slot in enumerate(slots)}
				i += len(slots)

				routedgrad[node.name] = d

		return routedgrad


	def gradShapeFrom(self, outshapes, shapes):
		shape = self.buildOutGradShape(outshapes, shapes)

		inshape = self.routeInGrad(self.module.gradShapeFrom(shape))
		shapes[self.name] = inshape


	def buildOutGradShape(self, outshapes, shapes):
		if len(self.fwds) == 0:
			return outshapes[self.name]

		shape = [None for _ in range(len(self.data) if isinstance(self.data, list) else 1)]

		for node, slots in self.fwds:
			if slots is not None:
				for slot in slots:
					shape[slot] = shapes[node.name][self.name][slot]

			else:
				for i, sh in enumerate(shapes[node.name][self.name]):
					shape[i] = sh

		if len(shape) == 1:
			shape = shape[0]

		return shape


	def reset(self):
		self.clearTraverse()

		self.data = None
		self.grad = None

		self.module.reset()


	def clearTraverse(self):
		self.fwdVisited = False
		self.bwdVisited = False


	def __str__(self):
		return "Node %s (name: %s)" % (type(self.module), self.name)


	@staticmethod
	def traverseForward(node, func, *args):
		while True:
			if node.fwdVisited:
				return

			if not all(bwd[0].fwdVisited for bwd in node.bwds):
				return

			func(node, *args)
			node.fwdVisited = True

			if len(node.fwds) == 1:
				node, _ = node.fwds[0]
				continue

			else:
				for n, _ in node.fwds:
					n.traverseForward(n, func, *args)

				break


	@staticmethod
	def traverseBackward(node, func, *args):
		while True:
			if node.bwdVisited:
				return

			if not all(fwd[0].bwdVisited for fwd in node.fwds):
				return

			func(node, *args)
			node.bwdVisited = True

			if len(node.bwds) == 1:
				node, _ = node.bwds[0]
				continue

			else:
				for n, _ in node.bwds:
					n.traverseBackward(n, func, *args)

				break
