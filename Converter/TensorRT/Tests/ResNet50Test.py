from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.ResNet import loadResNet

from PuzzleLib.Converter.Examples.Common import loadResNetSample, loadLabels

from PuzzleLib.Converter.TensorRT.Tests.Common import scoreModels, benchModels
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine


def main():
	net = loadResNet(modelpath="../../TestData/ResNet-50-model.hdf", layers="50")

	data = gpuarray.to_gpu(loadResNetSample(net, "../../TestData/tarantula.jpg"))
	labels = loadLabels(synpath="../../TestData/synsets.txt", wordpath="../../TestData/synset_words.txt")

	engine = buildRTEngine(net, inshape=data.shape, savepath="../TestData")

	scoreModels(net, engine, data, labels)
	benchModels(net, engine, data)


if __name__ == "__main__":
	main()
