from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.VGG import loadVGG

from PuzzleLib.Converter.Examples.Common import loadVGGSample, loadLabels, showLabelResults


def main():
	vgg16Test()
	vgg19Test()


def vgg16Test():
	net = loadVGG(modelpath="../TestData/VGG_ILSVRC_16_layers.hdf", layers="16")

	sample = loadVGGSample("../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


def vgg19Test():
	net = loadVGG(modelpath="../TestData/VGG_ILSVRC_19_layers.hdf", layers="19")

	sample = loadVGGSample("../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


if __name__ == "__main__":
	main()
