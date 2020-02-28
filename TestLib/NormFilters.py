from PuzzleLib.Backend import gpuarray

from PuzzleLib import Visual
from PuzzleLib.Modules import SubtractMean, LCN


def main():
	subtractMean = SubtractMean(size=7)
	lcn = LCN(N=7)

	img = gpuarray.to_gpu(Visual.loadImage("../TestData/Bench.png"))

	subtractMean(img)
	Visual.showImage(subtractMean.data.get(), "../TestData/ResultSubtractNorm.png")

	lcn(img)
	Visual.showImage(lcn.data.get(), "../TestData/ResultLCN.png")


if __name__ == "__main__":
	main()
