import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers import Sequential, Parallel
from PuzzleLib.Modules import Conv2D, Activation, relu, MaxPool2D, Replicate, Identity, Concat

from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngineFromCaffe, DataType
from PuzzleLib.Converter.TensorRT.Tests.Common import benchModels


def loadNet(modelpath=None, name="OpenPoseFaceNet"):
	net = Sequential(name=name)

	net.append(Conv2D(3, 64, 3, pad=1, name="conv1_1"))
	net.append(Activation(relu, name="conv1_1_re"))
	net.append(Conv2D(64, 64, 3, pad=1, name="conv1_2"))
	net.append(Activation(relu, name="conv1_2_re"))

	net.append(MaxPool2D(2, 2, name="pool1"))

	net.append(Conv2D(64, 128, 3, pad=1, name="conv2_1"))
	net.append(Activation(relu, name="conv2_1_re"))
	net.append(Conv2D(128, 128, 3, pad=1, name="conv2_2"))
	net.append(Activation(relu, name="conv2_2_re"))

	net.append(MaxPool2D(2, 2, name="pool2"))

	net.append(Conv2D(128, 256, 3, pad=1, name="conv3_1"))
	net.append(Activation(relu, name="conv3_1_re"))
	net.append(Conv2D(256, 256, 3, pad=1, name="conv3_2"))
	net.append(Activation(relu, name="conv3_2_re"))
	net.append(Conv2D(256, 256, 3, pad=1, name="conv3_3"))
	net.append(Activation(relu, name="conv3_3_re"))
	net.append(Conv2D(256, 256, 3, pad=1, name="conv3_4"))
	net.append(Activation(relu, name="conv3_4_re"))

	net.append(MaxPool2D(2, 2, name="pool3"))

	net.append(Conv2D(256, 512, 3, pad=1, name="conv4_1"))
	net.append(Activation(relu, name="conv4_1_re"))
	net.append(Conv2D(512, 512, 3, pad=1, name="conv4_2"))
	net.append(Activation(relu, name="conv4_2_re"))
	net.append(Conv2D(512, 512, 3, pad=1, name="conv4_3"))
	net.append(Activation(relu, name="conv4_3_re"))
	net.append(Conv2D(512, 512, 3, pad=1, name="conv4_4"))
	net.append(Activation(relu, name="conv4_4_re"))

	net.append(Conv2D(512, 512, 3, pad=1, name="conv5_1"))
	net.append(Activation(relu, name="conv5_1_re"))
	net.append(Conv2D(512, 512, 3, pad=1, name="conv5_2"))
	net.append(Activation(relu, name="conv5_2_re"))

	net.append(Conv2D(512, 128, 3, pad=1, name="conv5_3_CPM"))
	net.append(Activation(relu, name="conv5_3_CPM_re"))

	net.append(Replicate(2))

	shortcut0 = Sequential()
	shortcut0.append(Identity())

	branch0 = Sequential()
	branch0.append(Replicate(2))

	shortcut1 = Sequential()
	shortcut1.append(Identity())

	branch1 = Sequential()
	branch1.append(Replicate(2))

	shortcut2 = Sequential()
	shortcut2.append(Identity())

	branch2 = Sequential()
	branch2.append(Replicate(2))

	shortcut3 = Sequential()
	shortcut3.append(Identity())

	branch3 = Sequential()
	branch3.append(Replicate(2))

	shortcut4 = Sequential()
	shortcut4.append(Identity())

	branch4 = Sequential()
	branch4.append(Conv2D(128, 512, 1, pad=0, name="conv6_1_CPM"))
	branch4.append(Activation(relu, name="conv6_1_CPM_re"))
	branch4.append(Conv2D(512, 71, 1, pad=0, name="conv6_2_CPM"))

	branches = [branch4, branch3, branch2, branch1, branch0, net]
	shortcuts = [shortcut4, shortcut3, shortcut2, shortcut1, shortcut0, None]

	for branchIdx, branch in enumerate(branches):
		if branchIdx == 0:
			continue

		branch.append(Parallel().append(branches[branchIdx - 1]).append(shortcuts[branchIdx - 1]))
		branch.append(Concat(name="features_in_stage_%d" % (branchIdx + 1), axis=1))

		branch.append(Conv2D(199, 128, 7, pad=3, name="Mconv1_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv1_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 128, 7, pad=3, name="Mconv2_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv2_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 128, 7, pad=3, name="Mconv3_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv3_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 128, 7, pad=3, name="Mconv4_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv4_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 128, 7, pad=3, name="Mconv5_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv5_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 128, 1, pad=0, name="Mconv6_stage%d" % (branchIdx + 1)))
		branch.append(Activation(relu, name="Mconv6_stage%d_re" % (branchIdx + 1)))
		branch.append(Conv2D(128, 71, 1, pad=0, name="Mconv7_stage%d" % (branchIdx + 1)))

	if modelpath is not None:
		net.load(modelpath, assumeUniqueNames=True, name=name)
		net.evalMode()

	return net


def main():
	inshape = (1, 3, 368, 368)

	net = loadNet("../TestData/pose_iter_116000.hdf")
	net.optimizeForShape(inshape)

	outshape = net.dataShapeFrom(inshape)

	engine = buildRTEngineFromCaffe(
		("../TestData/pose_deploy.prototxt", "../TestData/pose_iter_116000.caffemodel"),
		inshape=inshape, outshape=outshape, outlayers=["net_output"], dtype=DataType.float32, savepath="../TestData"
	)

	data = gpuarray.to_gpu(np.random.randn(*inshape).astype(np.float32))

	netData = net(data).get()
	engineData = engine(data).get()

	assert np.allclose(netData, engineData)
	benchModels(net, engine, data)


if __name__ == "__main__":
	main()
