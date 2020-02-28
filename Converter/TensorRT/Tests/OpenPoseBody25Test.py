import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngineFromCaffe, DataType


def main():
	inshape = (1, 3, 16, 16)
	outshape = (1, 78, 2, 2)

	engine = buildRTEngineFromCaffe(
		("../TestData/pose_deploy.prototxt", "../TestData/pose_iter_584000.caffemodel"),
		inshape=inshape, outshape=outshape, outlayers=["net_output"], dtype=DataType.float32, savepath="../TestData"
	)

	data = gpuarray.to_gpu(np.random.randn(*inshape).astype(np.float32))
	engine(data)


if __name__ == "__main__":
	main()
