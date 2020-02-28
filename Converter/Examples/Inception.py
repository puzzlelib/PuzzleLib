from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.Inception import loadInceptionBN, loadInceptionV3

from PuzzleLib.Converter.Examples.Common import loadVGGSample, loadLabels, loadV3Labels, showLabelResults


def main():
	inceptionBNTest()
	inceptionV3Test()


def inceptionBNTest():
	net = loadInceptionBN(modelpath="../TestData/Inception-BN-0126.hdf")

	sample = loadVGGSample("../TestData/tarantula.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


def inceptionV3Test():
	net = loadInceptionV3(modelpath="../TestData/Inception-7-0001.hdf")

	sample = loadVGGSample("../TestData/tarantula.jpg", shape=(299, 299), normalize=True)
	labels = loadV3Labels(filename="../TestData/synset_inception_v3.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header=net.name)


if __name__ == "__main__":
	main()
