from PuzzleLib.Cuda.Kernels.MatVec import backendTest


def unittest():
	from PuzzleLib.Hip import Backend
	backendTest(Backend)


if __name__ == "__main__":
	unittest()
