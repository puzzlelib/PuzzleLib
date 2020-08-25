import numpy as np
from PuzzleLib.Cuda.Backend import getBackend


def main():
	datashape = (128, 32, 64, 64)
	Wshape = (64, 32, 11, 11)

	stride, pad, dilation, groups = 1, 0, 1, datashape[1] // Wshape[1]
	timeConv(getBackend(initmode=1), datashape, Wshape, np.float32, stride, pad, dilation, groups)


def timeConv(backend, datashape, Wshape, dtype, stride, pad, dilation, groups):
	fwdResults, bwdDataResults, bwdFilterResults = backend.convNdbenchmark(
		datashape, Wshape, dtype, stride, pad, dilation, groups
	)

	print("Forward results:")
	for res in fwdResults:
		print(res)

	print("\nBackward filter results:")
	for res in bwdFilterResults:
		print(res)

	print("\nBackward data results:")
	for res in bwdDataResults:
		print(res)


if __name__ == "__main__":
	main()
