import os

import numpy as np
import onnx, onnx.shape_inference

from PuzzleLib.Models.Nets.ResNet import loadResNet

from PuzzleLib.Containers.Container import Container
from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel
from PuzzleLib.Containers.Graph import Graph

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
from PuzzleLib.Modules.Activation import Activation, relu, leakyRelu
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Add import Add
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Dropout import Dropout
from PuzzleLib.Modules.MulAddConst import MulAddConst
from PuzzleLib.Modules.BatchNorm import BatchNorm
from PuzzleLib.Modules.SoftMax import SoftMax
from PuzzleLib.Modules.Split import Split
from PuzzleLib.Modules.Upsample2D import Upsample2D


class ONNXExporter:
	def __init__(self, validate=True, exportWeights=True):
		self.validate = validate
		self.exportWeights = exportWeights

		self.nodes = []
		self.initializer = []


	def export(self, net, inshape, savepath):
		outshape = net.dataShapeFrom(inshape)

		inshape = [inshape] if not isinstance(inshape, list) else inshape
		outshape = [outshape] if not isinstance(outshape, list) else outshape

		inputs = ["data_%s" % i for i in range(len(inshape))]
		outputs = self.convertModule(net, net.name, inputs)

		inputs = [
			onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, inshape[i])
			for i, name in enumerate(inputs)
		]

		inputs.extend(
			onnx.helper.make_tensor_value_info(init.name, init.data_type, init.dims) for init in self.initializer
		)

		outputs = [
			onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, outshape[i])
			for i, name in enumerate(outputs)
		]

		graph = onnx.helper.make_graph(self.nodes, net.name, inputs, outputs, initializer=self.initializer)
		model = onnx.helper.make_model(graph, producer_name="puzzlelib")

		if self.validate:
			onnx.checker.check_model(model)

		if not self.exportWeights:
			model.graph.ClearField("initializer")

		model = onnx.shape_inference.infer_shapes(model)
		onnx.save_model(model, os.path.join(savepath, "%s.onnx" % net.name))

		return model


	def convertModule(self, module, fullname, inputs):
		if isinstance(module, Container):
			if isinstance(module, Sequential):
				return self.convertSequential(module, fullname, inputs)

			elif isinstance(module, Parallel):
				return self.convertParallel(module, fullname, inputs)

			elif isinstance(module, Graph):
				return self.convertGraph(module, fullname, inputs)

			else:
				raise NotImplementedError(module.__class__.__name__)

		else:
			if isinstance(module, Add):
				return self.convertAdd(fullname, inputs)

			elif isinstance(module, Concat):
				return self.convertConcat(module, fullname, inputs)

			assert len(inputs) == 1
			inp = inputs[0]

			if isinstance(module, Conv2D):
				return self.convertConv(module, fullname, inp)

			elif isinstance(module, (BatchNorm, BatchNorm2D)):
				return self.convertBatchNorm(module, fullname, inp)

			elif isinstance(module, Activation):
				return self.convertActivation(module, fullname, inp)

			elif isinstance(module, (Identity, Dropout)):
				return self.convertIdentity(inp)

			elif isinstance(module, (MaxPool2D, AvgPool2D)):
				return self.convertPool(module, fullname, inp)

			elif isinstance(module, Flatten):
				return self.convertFlatten(fullname, inp)

			elif isinstance(module, Linear):
				return self.convertLinear(module, fullname, inp)

			elif isinstance(module, SoftMax):
				return self.convertSoftmax(fullname, inp)

			elif isinstance(module, Replicate):
				return self.convertReplicate(module, inp)

			elif isinstance(module, MulAddConst):
				return self.convertMulAddConst(module, fullname, inp)

			elif isinstance(module, Split):
				return self.convertSplit(module, fullname, inp)

			elif isinstance(module, Upsample2D):
				return self.convertUpsample2D(module, fullname, inp)

			else:
				raise NotImplementedError(module.__class__.__name__)


	def convertSequential(self, seq, fullname, inputs):
		for child in seq.graph:
			name = "%s.%s" % (fullname, child.name)
			inputs = self.convertModule(child, name, inputs)

		return inputs


	def convertParallel(self, parallel, fullname, inputs):
		assert len(inputs) == len(parallel.graph)

		outputs = []
		for i, child in enumerate(parallel.graph):
			name = "%s.%s" % (fullname, child.name)
			outputs.append(self.convertModule(child, name, [inputs[i]])[0])

		return outputs


	def convertNode(self, node, fullname, inputs, nodes):
		name = None if node.name is None else "%s.%s" % (fullname, node.name)
		inputs = [inputs[node.name]] if len(node.bwds) == 0 else [nodes[output.name] for output, _ in node.bwds]

		outputs = self.convertModule(node.module, name, inputs)
		assert len(outputs) == 1

		nodes[node.name] = outputs[0]


	def convertGraph(self, graph, fullname, inputs):
		assert len(inputs) == len(graph.inputs)

		nodes = {}
		inputs = {node.name: inputs[i] for i, node in enumerate(graph.inputs)}

		for i, inp in enumerate(graph.inputs):
			inp.traverseForward(inp, self.convertNode, fullname, inputs, nodes)

		graph.reset()
		outputs = [nodes[output.name] for output in graph.outputs]

		return outputs


	def convertAdd(self, fullname, inputs):
		assert len(inputs) == 2

		self.nodes.append(onnx.helper.make_node(
			"Add", inputs=inputs, outputs=[fullname]
		))

		return [fullname]


	def convertConcat(self, module, fullname, inp):
		self.nodes.append(onnx.helper.make_node(
			"Concat", inputs=inp, outputs=[fullname],
			axis=module.axis
		))

		return [fullname]


	def convertConv(self, module, fullname, inp):
		assert module.dilation == (1, 1) and module.groups == 1
		strides = module.stride

		wpad, hpad = module.pad
		pads = [wpad, hpad, wpad, hpad]

		Wname = "%s.W" % fullname
		W = module.W.get()

		self.initializer.append(onnx.helper.make_tensor(
			name=Wname, data_type=onnx.TensorProto.FLOAT, dims=W.shape, vals=W.flatten()
		))

		inputs = [inp, Wname]

		if module.useBias:
			biasname = "%s.b" % fullname
			bias = module.b.get()

			self.initializer.append(onnx.helper.make_tensor(
				name=biasname, data_type=onnx.TensorProto.FLOAT, dims=(bias.shape[1], ), vals=bias.flatten()
			))
			inputs.append(biasname)

		self.nodes.append(onnx.helper.make_node(
			"Conv", inputs=inputs, outputs=[fullname],
			pads=pads, strides=strides
		))

		return [fullname]


	def convertBatchNorm(self, module, fullname, inp):
		scalename, biasname = "%s.scale" % fullname, "%s.bias" % fullname
		meanname, varname = "%s.mean" % fullname, "%s.var" % fullname

		scale, bias = module.scale.get().flatten(), module.bias.get().flatten()
		mean, var = module.mean.get().flatten(), module.var.get().flatten()

		for name, tensor in [(scalename, scale), (biasname, bias), (meanname, mean), (varname, var)]:
			self.initializer.append(onnx.helper.make_tensor(
				name=name, data_type=onnx.TensorProto.FLOAT, dims=tensor.shape, vals=tensor
			))

		self.nodes.append(onnx.helper.make_node(
			"BatchNormalization", inputs=[inp, scalename, biasname, meanname, varname], outputs=[fullname],
			epsilon=module.epsilon
		))

		return [fullname]


	def convertActivation(self, module, fullname, inp):
		actType = module.getBlueprint()["scheme"]["activation"]
		assert actType in {relu, leakyRelu}

		if actType == relu:
			typ = "Relu"
			attrs = {}

		else:
			typ = "LeakyRelu"
			attrs = {"alpha": module.args[0]}

		self.nodes.append(onnx.helper.make_node(typ, inputs=[inp], outputs=[fullname], **attrs))
		return [fullname]


	@classmethod
	def convertIdentity(cls, inp):
		return [inp]


	def convertPool(self, module, fullname, inp):
		typ = {
			MaxPool2D: "MaxPool",
			AvgPool2D: "AveragePool"
		}[type(module)]

		strides = module.stride

		wpad, hpad = module.pad
		pads = [wpad, hpad, wpad, hpad]

		self.nodes.append(onnx.helper.make_node(
			typ, inputs=[inp], outputs=[fullname],
			kernel_shape=module.size, pads=pads, strides=strides
		))

		return [fullname]


	def convertFlatten(self, fullname, inp):
		self.nodes.append(onnx.helper.make_node(
			"Flatten", inputs=[inp], outputs=[fullname],
			axis=1
		))

		return [fullname]


	def convertLinear(self, module, fullname, inp):
		Wname = "%s.W" % fullname
		W = module.W.get()

		self.initializer.append(onnx.helper.make_tensor(
			name=Wname, data_type=onnx.TensorProto.FLOAT, dims=W.shape, vals=W.flatten()
		))

		mulname = "%s.mul" % fullname

		self.nodes.append(onnx.helper.make_node(
			"MatMul", inputs=[inp, Wname], outputs=[mulname]
		))

		if module.useBias:
			biasname = "%s.b" % fullname
			bias = module.b.get()

			self.initializer.append(onnx.helper.make_tensor(
				name=biasname, data_type=onnx.TensorProto.FLOAT, dims=bias.shape, vals=bias
			))

			self.nodes.append(onnx.helper.make_node(
				"Add", inputs=[mulname, biasname], outputs=[fullname]
			))

		else:
			fullname = mulname

		return [fullname]


	def convertSoftmax(self, fullname, inp):
		self.nodes.append(onnx.helper.make_node(
			"Softmax", inputs=[inp], outputs=[fullname],
			axis=1
		))

		return [fullname]


	@classmethod
	def convertReplicate(cls, module, inp):
		return [inp] * module.times


	def convertMulAddConst(self, module, fullname, inp):
		aname, bname = "%s.a" % fullname, "%s.b" % fullname
		a, b = np.array([module.a], dtype=np.float32), np.array([module.b], dtype=np.float32)

		for name, tensor in [(aname, a), (bname, b)]:
			self.initializer.append(onnx.helper.make_tensor(
				name=name, data_type=onnx.TensorProto.FLOAT, dims=tensor.shape, vals=tensor
			))

		mulname = "%s.mul" % fullname

		self.nodes.append(onnx.helper.make_node(
			"Mul", inputs=[inp, aname], outputs=[mulname]
		))

		self.nodes.append(onnx.helper.make_node(
			"Add", inputs=[mulname, bname], outputs=[fullname]
		))

		return [fullname]


	def convertSplit(self, module, fullname, inp):
		outputs = ["%s_%s" % (fullname, i) for i in range(len(module.sections))]

		self.nodes.append(onnx.helper.make_node(
			"Split", inputs=inp, outputs=outputs, axis=module.axis, split=module.sections
		))

		return outputs


	def convertUpsample2D(self, module, fullname, inp):
		assert module.mode == "nearest"

		roiname = "%s.roi" % fullname
		roi = np.array([], dtype=np.float32)

		self.initializer.append(onnx.helper.make_tensor(
			name=roiname, data_type=onnx.TensorProto.FLOAT, dims=roi.shape, vals=roi
		))

		scalename = "%s.scales" % fullname
		scales = np.array([1.0, 1.0, module.scale, module.scale], dtype=np.float32)

		self.initializer.append(onnx.helper.make_tensor(
			name=scalename, data_type=onnx.TensorProto.FLOAT, dims=scales.shape, vals=scales
		))

		self.nodes.append(onnx.helper.make_node(
			"Resize", inputs=[inp, roiname, scalename], outputs=[fullname], mode="nearest"
		))

		return [fullname]


def unittest():
	net = loadResNet(modelpath="../TestData/ResNet-50-model.hdf", layers="50")
	ONNXExporter().export(net, inshape=(1, 3, 224, 224), savepath="../TestData/")


if __name__ == "__main__":
	unittest()
