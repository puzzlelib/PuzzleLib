from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules import SubtractMean, LCN
from PuzzleLib.Visual import loadImage, showImage


def main():
	subtractMean = SubtractMean(size=7)
	lcn = LCN(N=7)

	img = gpuarray.to_gpu(loadImage("../TestData/Bench.png"))

	subtractMean(img)
	showImage(subtractMean.data.get(), "../TestData/ResultSubtractNorm.png")

	lcn(img)
	showImage(lcn.data.get(), "../TestData/ResultLCN.png")


if __name__ == "__main__":
	main()
