import numpy as np
from PIL import Image

from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.ResNet import loadResNet

from PuzzleLib.Converter.Examples.Common import loadResNetSample, loadLabels

from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngineFromCaffe, buildRTEngineFromOnnx, DataType
from PuzzleLib.Converter.TensorRT.Tests.Common import printResults


def preprocessCaffe2Onnx(img):
	mean = np.array([0.485, 0.456, 0.406])
	stddev = np.array([0.229, 0.224, 0.225])

	normdata = np.zeros(img.shape).astype(np.float32)

	for i in range(img.shape[0]):
		normdata[i, :, :] = (img[i, :, :] / 255 - mean[i]) / stddev[i]

	return normdata


def main():
	inshape = (1, 3, 224, 224)

	net = loadResNet(modelpath="../../TestData/ResNet-50-model.hdf", layers="50")
	outshape = net.dataShapeFrom(inshape)

	caffeengine = buildRTEngineFromCaffe(
		("../TestData/ResNet-50-deploy.prototxt", "../TestData/ResNet-50-model.caffemodel"),
		inshape=inshape, outshape=outshape, outlayers=["prob"], dtype=DataType.float32, savepath="../TestData"
	)

	onnxengine = buildRTEngineFromOnnx(
		"../TestData/resnet50.onnx", inshape=inshape, outshape=outshape, dtype=DataType.float32, savepath="../TestData"
	)

	data = gpuarray.to_gpu(loadResNetSample(net, "../../TestData/tarantula.jpg"))
	labels = loadLabels(synpath="../../TestData/synsets.txt", wordpath="../../TestData/synset_words.txt")

	netData = net(data).get()
	caffeData = caffeengine(data).get()

	data = np.moveaxis(np.array(Image.open("../../TestData/tarantula.jpg"), dtype=np.float32), 2, 0)
	data = gpuarray.to_gpu(preprocessCaffe2Onnx(data)[np.newaxis, ...])

	onnxData = onnxengine(data).get()

	printResults(netData, labels, "Net")
	printResults(caffeData, labels, "Caffe")
	printResults(onnxData, labels, "Onnx")


if __name__ == "__main__":
	main()
