from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.ResNet import loadResNet

from PuzzleLib.Converter.Examples.Common import loadResNetSample, loadLabels, showLabelResults


def main():
	resNet50Test()
	resNet101Test()
	resNet152Test()


def resNet50Test():
	net = loadResNet(modelpath="../TestData/ResNet-50-model.hdf", layers="50")

	sample = loadResNetSample(net, "../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


def resNet101Test():
	net = loadResNet(modelpath="../TestData/ResNet-101-model.hdf", layers="101")

	sample = loadResNetSample(net, "../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


def resNet152Test():
	net = loadResNet(modelpath="../TestData/ResNet-152-model.hdf", layers="152")

	sample = loadResNetSample(net, "../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


if __name__ == "__main__":
	main()
