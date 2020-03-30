from PuzzleLib.Cuda.Kernels.CTC import CTCModule, backendTest


class HipCTCModule(CTCModule):
	@staticmethod
	def generateConfig(backend):
		return [
			(backend.warpSize, 1),
			(backend.warpSize * 2, 1),
			(backend.warpSize, 3),
			(backend.warpSize * 2, 2),
			(backend.warpSize, 6),
			(backend.warpSize * 2, 4),
			(backend.warpSize, 9),
			(backend.warpSize * 2, 6),
			(backend.warpSize * 2, 9),
			(backend.warpSize * 2, 10)
		]


def unittest():
	from PuzzleLib.Hip import Backend
	backendTest(Backend, HipCTCModule)


if __name__ == "__main__":
	unittest()
