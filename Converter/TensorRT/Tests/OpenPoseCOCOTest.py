import numpy as np

from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.MaxPool2D import MaxPool2D

from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel

from PuzzleLib.Converter.TensorRT.Tests.Common import benchModels
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine, buildRTEngineFromCaffe, DataType


def buildSmallBlock(inplace=True):
	block = Sequential()

	block.append(Replicate(3))

	left = buildSmallBranch(inplace=inplace, num=1)
	right = buildSmallBranch(inplace=inplace, num=2)

	shortcut = Sequential().append(Identity())

	block.append(Parallel().append(left).append(right).append(shortcut))
	block.append(Concat(axis=1, name="concat_stage2"))

	return block


def buildSmallBranch(inplace=True, num=1):
	branch = Sequential()

	branch.append(Conv2D(128, 128, 3, pad=1, initscheme="none", name="conv5_1_CPM_L%d" % num))
	branch.append(Activation(relu, inplace=inplace, name="relu5_1_CPM_L%d" % num))

	branch.append(Conv2D(128, 128, 3, pad=1, initscheme="none", name="conv5_2_CPM_L%d" % num))
	branch.append(Activation(relu, inplace=inplace, name="relu5_2_CPM_L%d" % num))

	branch.append(Conv2D(128, 128, 3, pad=1, initscheme="none", name="conv5_3_CPM_L%d" % num))
	branch.append(Activation(relu, inplace=inplace, name="relu5_3_CPM_L%d" % num))

	branch.append(Conv2D(128, 512, 1, initscheme="none", name="conv5_4_CPM_L%d" % num))
	branch.append(Activation(relu, inplace=inplace, name="relu5_4_CPM_L%d" % num))

	branch.append(Conv2D(512, 19 * (3 - num), 1, initscheme="none", name="conv5_5_CPM_L%d" % num))

	return branch


def buildBranch(inmaps=185, inplace=True, num=1, stage=2):
	branch = Sequential()

	branch.append(Conv2D(inmaps, 128, 7, pad=3, initscheme="none", name="Mconv1_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu1_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 128, 7, pad=3, initscheme="none", name="Mconv2_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu2_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 128, 7, pad=3, initscheme="none", name="Mconv3_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu3_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 128, 7, pad=3, initscheme="none", name="Mconv4_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu4_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 128, 7, pad=3, initscheme="none", name="Mconv5_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu5_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 128, 1, initscheme="none", name="Mconv6_stage%d_L%d" % (stage, num)))
	branch.append(Activation(relu, inplace=inplace, name="Mrelu6_stage%d_L%d" % (stage, num)))

	branch.append(Conv2D(128, 19 * (3 - num), 1, initscheme="none", name="Mconv7_stage%d_L%d" % (stage, num)))

	return branch


def buildBall(stage=2, inplace=True):
	ball = Sequential()

	ball.append(Replicate(2))

	left = buildBranch(stage=stage, num=1, inplace=inplace)
	right = buildBranch(stage=stage, num=2, inplace=inplace)

	ball.append(Parallel().append(left).append(right))

	ball.append(Concat(axis=1))

	return ball


def buildBigBlock(stage=2, prenet=None, inplace=True):
	block = Sequential()

	block.append(Replicate(2))

	shortcut = Sequential().append(Identity())

	if prenet is None:
		ball = buildBall(stage=stage, inplace=inplace)
	else:
		ball = prenet
		ball.extend(buildBall(stage=stage, inplace=inplace))

	block.append(Parallel().append(ball).append(shortcut))
	block.append(Concat(axis=1, name="concat_stage%d" % (stage+1)))

	return block


def loadNet(name="", inplace=True, modelpath=None):
	net = Sequential(name)

	net.append(Conv2D(3, 64, 3, pad=1, initscheme="none", name="conv1_1"))
	net.append(Activation(relu, name="relu1_1", inplace=inplace))

	net.append(Conv2D(64, 64, 3, pad=1, initscheme="none", name="conv1_2"))
	net.append(Activation(relu, name="relu1_2", inplace=inplace))

	net.append(MaxPool2D(name="pool1_stage1"))

	net.append(Conv2D(64, 128, 3, pad=1, initscheme="none", name="conv2_1"))
	net.append(Activation(relu, name="relu2_1", inplace=inplace))

	net.append(Conv2D(128, 128, 3, pad=1, initscheme="none", name="conv2_2"))
	net.append(Activation(relu, name="relu2_2", inplace=inplace))

	net.append(MaxPool2D(name="pool2_stage1"))

	net.append(Conv2D(128, 256, 3, pad=1, initscheme="none", name="conv3_1"))
	net.append(Activation(relu, name="relu3_1", inplace=inplace))

	net.append(Conv2D(256, 256, 3, pad=1, initscheme="none", name="conv3_2"))
	net.append(Activation(relu, name="relu3_2", inplace=inplace))

	net.append(Conv2D(256, 256, 3, pad=1, initscheme="none", name="conv3_3"))
	net.append(Activation(relu, name="relu3_3", inplace=inplace))

	net.append(Conv2D(256, 256, 3, pad=1, initscheme="none", name="conv3_4"))
	net.append(Activation(relu, name="relu3_4", inplace=inplace))

	net.append(MaxPool2D(name="pool3_stage1"))

	net.append(Conv2D(256, 512, 3, pad=1, initscheme="none", name="conv4_1"))
	net.append(Activation(relu, name="relu4_1", inplace=inplace))

	net.append(Conv2D(512, 512, 3, pad=1, initscheme="none", name="conv4_2"))
	net.append(Activation(relu, name="relu4_2", inplace=inplace))

	net.append(Conv2D(512, 256, 3, pad=1, initscheme="none", name="conv4_3_CPM"))
	net.append(Activation(relu, name="relu4_3_CPM"))

	net.append(Conv2D(256, 128, 3, pad=1, initscheme="none", name="conv4_4_CPM"))
	net.append(Activation(relu, name="relu4_4_CPM"))

	block2 = buildSmallBlock(inplace=inplace)
	block3 = buildBigBlock(stage=2, prenet=block2, inplace=inplace)
	block4 = buildBigBlock(stage=3, prenet=block3, inplace=inplace)
	block5 = buildBigBlock(stage=4, prenet=block4, inplace=inplace)
	block6 = buildBigBlock(stage=5, prenet=block5, inplace=inplace)

	net.extend(block6)
	net.append(Replicate(2))

	net.append(Parallel().append(
		buildBranch(stage=6, num=2, inplace=inplace)
	).append(
		buildBranch(stage=6, num=1, inplace=inplace))
	)

	net.append(Concat(axis=1))

	if modelpath is not None:
		net.load(modelpath, assumeUniqueNames=True)

	return net


def main():
	inshape = (1, 3, 368, 368)

	net = loadNet(inplace=False, modelpath="../TestData/pose_iter_440000.hdf")
	outshape = net.dataShapeFrom(inshape)

	pzlEngine = buildRTEngine(net, inshape=inshape, dtype=DataType.float32, savepath="../TestData")
	caffeEngine = buildRTEngineFromCaffe(
		("../TestData/pose_deploy_linevec.prototxt", "../TestData/pose_iter_440000.caffemodel"),
		inshape=inshape, outshape=outshape, outlayers=["net_output"], dtype=DataType.float32, savepath="../TestData"
	)

	data = gpuarray.to_gpu(np.random.randn(*inshape).astype(np.float32))

	pzlData = pzlEngine(data)
	caffeData = caffeEngine(data)

	assert np.allclose(pzlData.get(), caffeData.get(), atol=1e-7)
	benchModels(pzlEngine, caffeEngine, data, lognames=("puzzle", "caffe "))


if __name__ == "__main__":
	main()
