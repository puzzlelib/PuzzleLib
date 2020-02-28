from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets import loadNiNImageNet

from PuzzleLib.Converter.Examples.Common import loadSample, loadLabels, showLabelResults


def main():
	net = loadNiNImageNet(modelpath="../TestData/nin_imagenet.hdf")

	sample = loadSample("../TestData/barometer.jpg")
	labels = loadLabels(synpath="../TestData/synsets.txt", wordpath="../TestData/synset_words.txt")

	res = net(gpuarray.to_gpu(sample)).get().reshape(-1)
	showLabelResults(res, labels, header="NiN")


if __name__ == "__main__":
	main()
