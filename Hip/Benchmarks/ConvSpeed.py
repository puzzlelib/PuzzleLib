import numpy as np

from PuzzleLib import Config
from PuzzleLib.Cuda.Benchmarks.ConvSpeed import timeConv


def main():
	datashape = (128, 32, 64, 64)
	Wshape = (64, 32, 11, 11)

	stride, pad, dilation, groups = 1, 0, 1, datashape[1] // Wshape[1]

	from PuzzleLib.Hip.Backend import getBackend
	backend = getBackend(Config.deviceIdx, initmode=1)

	timeConv(backend, datashape, Wshape, np.float32, stride, pad, dilation, groups)


if __name__ == "__main__":
	main()
