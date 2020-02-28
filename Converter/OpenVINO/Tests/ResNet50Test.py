from PuzzleLib import Config

Config.backend = Config.Backend.intel
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.ResNet import loadResNet

from PuzzleLib.Converter.Examples.Common import loadResNetSample, loadLabels

from PuzzleLib.Converter.OpenVINO.Tests.Common import scoreModels, benchModels
from PuzzleLib.Converter.OpenVINO.BuildVINOEngine import buildVINOEngine


def main():
	net = loadResNet(modelpath="../../TestData/ResNet-50-model.hdf", layers="50")

	data = gpuarray.to_gpu(loadResNetSample(net, "../../TestData/tarantula.jpg"))
	labels = loadLabels(synpath="../../TestData/synsets.txt", wordpath="../../TestData/synset_words.txt")

	engine = buildVINOEngine(net, inshape=data.shape, savepath="../TestData")

	scoreModels(net, engine, data, labels)
	benchModels(net, engine, data)


if __name__ == "__main__":
	main()
