import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.SoftMax import SoftMax
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.ToList import ToList


def loadInceptionBN(modelpath, actInplace=False, bnInplace=False, initscheme="none", name="Inception-BN-0126"):
	net = Sequential(name=name)

	net.append(Conv2D(3, 64, 7, stride=2, pad=3, useBias=False, initscheme=initscheme, name="conv_1"))
	net.append(BatchNorm2D(64, inplace=bnInplace, name="bn_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu_1"))

	net.append(MaxPool2D(3, 2, pad=1, name="pool_1"))

	net.append(Conv2D(64, 64, 1, useBias=False, initscheme=initscheme, name="conv_2_red"))
	net.append(BatchNorm2D(64, inplace=bnInplace, name="bn_2_red"))
	net.append(Activation(relu, inplace=actInplace, name="relu_2_red"))

	net.append(Conv2D(64, 192, 3, pad=1, useBias=False, initscheme=initscheme, name="conv_2"))
	net.append(BatchNorm2D(192, inplace=bnInplace, name="bn_2"))
	net.append(Activation(relu, inplace=actInplace, name="relu_2"))

	net.append(MaxPool2D(3, 2, pad=1, name="pool_2"))

	act, bn = actInplace, bnInplace
	net.extend(bnBlock(192, [64], [64, 64], [64, 96, 96], [32], act=act, bn=bn, scheme=initscheme, name="3a"))
	net.extend(bnBlock(256, [64], [64, 96], [64, 96, 96], [64], act=act, bn=bn, scheme=initscheme, name="3b"))
	net.extend(bnShrinkBlock(320, [128, 160], [64, 96, 96], bn=bn, act=act, scheme=initscheme, name="3c"))

	net.extend(bnBlock(576, [224], [64, 96], [96, 128, 128], [128], act=act, bn=bn, scheme=initscheme, name="4a"))
	net.extend(bnBlock(576, [192], [96, 128], [96, 128, 128], [128], act=act, bn=bn, scheme=initscheme, name="4b"))
	net.extend(bnBlock(576, [160], [128, 160], [128, 160, 160], [128], act=act,bn=bn, scheme=initscheme, name="4c"))
	net.extend(bnBlock(608, [96], [128,192], [160, 192, 192], [128], act=act, bn=bn, scheme=initscheme, name="4d"))
	net.extend(bnShrinkBlock(608, [128, 192], [192, 256, 256], act=act, bn=bn, scheme=initscheme, name="4e"))

	net.extend(bnBlock(1056, [352], [192, 320], [160,224,224], [128], act=act, bn=bn, scheme=initscheme, name="5a"))
	net.extend(bnBlock(1024, [352], [192, 320], [192,224,224], [128], act=act, bn=bn, scheme=initscheme, name="5b"))

	net.append(AvgPool2D(7, 1, name="global_pool"))
	net.append(Flatten(name="flatten"))
	net.append(Linear(1024, 1000, initscheme=initscheme, name="fc1"))
	net.append(SoftMax(name="softmax"))

	if modelpath is not None:
		net.load(modelpath, assumeUniqueNames=True)

	return net


def loadInceptionV3(modelpath, actInplace=False, bnInplace=False, initscheme="none", name="Inception-7-0001"):
	net = Sequential(name=name)

	net.append(Conv2D(3, 32, 3, stride=2, useBias=False, initscheme=initscheme, name="conv_conv2d"))
	net.append(BatchNorm2D(32, name="conv_batchnorm"))
	net.append(Activation(relu, inplace=actInplace, name="conv_relu"))

	net.append(Conv2D(32, 32, 3, useBias=False, initscheme=initscheme, name="conv_1_conv2d"))
	net.append(BatchNorm2D(32, name="conv_1_batchnorm"))
	net.append(Activation(relu, inplace=actInplace, name="conv_1_relu"))

	net.append(Conv2D(32, 64, 3, pad=1, useBias=False, initscheme=initscheme, name="conv_2_conv2d"))
	net.append(BatchNorm2D(64, name="conv_2_batchnorm"))
	net.append(Activation(relu, inplace=actInplace, name="conv_2_relu"))

	net.append(MaxPool2D(3, 2, name="pool"))

	net.append(Conv2D(64, 80, 1, useBias=False, initscheme=initscheme, name="conv_3_conv2d"))
	net.append(BatchNorm2D(80, name="conv_3_batchnorm"))
	net.append(Activation(relu, inplace=actInplace, name="conv_3_relu"))

	net.append(Conv2D(80, 192, 3, useBias=False, initscheme=initscheme, name="conv_4_conv2d"))
	net.append(BatchNorm2D(192, name="conv_4_batchnorm"))
	net.append(Activation(relu, inplace=actInplace, name="conv_4_relu"))

	net.append(MaxPool2D(3, 2, name="pool1"))

	act, bn = actInplace, bnInplace
	net.extend(bnBlock(192, [64], [48, 64], [64, 96, 96], [32], "mixed", act, bn, initscheme, 5, 2, "v3"))
	net.extend(bnBlock(256, [64], [48, 64], [64, 96, 96], [64], "mixed_1", act, bn, initscheme, 5, 2, "v3"))
	net.extend(bnBlock(288, [64], [48, 64], [64, 96, 96], [64], "mixed_2", act, bn, initscheme, 5, 2, "v3"))
	net.extend(bnShrinkBlock(288, [384], [64, 96, 96], "mixed_3", act, bn, initscheme, False, 0, "v3"))

	net.extend(factorBlock(768, [192], [128, 128, 192], [128,128,128,128,192], [192], "mixed_4", act, bn, initscheme))
	net.extend(factorBlock(768, [192], [160, 160, 192], [160,160,160,160,192], [192], "mixed_5", act, bn, initscheme))
	net.extend(factorBlock(768, [192], [160, 160, 192], [160,160,160,160,192], [192], "mixed_6", act, bn, initscheme))
	net.extend(factorBlock(768, [192], [192, 192, 192], [192,192,192,192,192], [192], "mixed_7", act, bn, initscheme))
	net.extend(v3ShrinkBlock(768, [192, 320], [192, 192, 192, 192], "mixed_8", act, bn, initscheme))

	net.extend(expandBlock(
		1280, [320], [384, 384, 384], [448, 384, 384, 384], [192], "mixed_9", act, bn, initscheme, pool="avg"
	))

	net.extend(expandBlock(
		2048, [320], [384, 384, 384], [448, 384, 384, 384], [192], "mixed_10", act, bn, initscheme, pool="max"
	))

	net.append(AvgPool2D(8, 1, name="global_pool"))
	net.append(Flatten(name="flatten"))
	net.append(Linear(2048, 1008, name="fc1"))
	net.append(SoftMax(name="softmax"))

	if modelpath is not None:
		net.load(modelpath, assumeUniqueNames=True)

	return net


def convBN(inmaps, outmaps, size, stride, pad, name, actInplace, bnInplace, scheme, typ="bn"):
	block = Sequential()

	if typ == "bn":
		names = ["conv_%s" % name, "bn_%s" % name, "relu_%s" % name]

	elif typ == "v3":
		names = ["%s_conv2d" % name, "%s_batchnorm" % name, "%s_relu" % name]

	else:
		raise ValueError("Unrecognized convBN type")

	block.append(Conv2D(inmaps, outmaps, size, stride, pad, useBias=False, initscheme=scheme, name=names[0]))
	block.append(BatchNorm2D(outmaps, inplace=bnInplace, name=names[1]))
	block.append(Activation(relu, inplace=actInplace, name=names[2]))

	return block


def pool2D(size, stride, pad, name):
	if "max" in name:
		return MaxPool2D(size, stride, pad)

	elif "avg" in name:
		return AvgPool2D(size, stride, pad)

	else:
		raise ValueError("Unrecognized pool type")


def tower(towername, names, maps, sizes, strides, pads, act, bn, scheme, typ="bn"):
	block = Sequential()

	lvlnames = ["%s_%s" % (towername, name) for name in names]

	for i, name in enumerate(lvlnames):
		if "pool" in name:
			block.append(pool2D(sizes[i], strides[i], pads[i], name=names[i]))

		else:
			act = False if i == len(names) - 1 else act
			block.extend(convBN(maps[i], maps[i+1], sizes[i], strides[i], pads[i], lvlnames[i], act, bn, scheme, typ))

	return block


def bnBlock(inmaps, b1m, b2m, b3m, b4m, name, act, bn, scheme, b2size=3, b2pad=1, typ="bn"):
	block = Sequential()

	if typ == "bn":
		b1towername, b1names = name, ["1x1"]
		b2towername, b2names = name, ["3x3_reduce","3x3"]
		b3towername, b3names = name, ["double_3x3_reduce", "double_3x3_0", "double_3x3_1"]
		b4towername, b4names = name, ["avg_pool", "proj"]

	elif typ == "v3":
		b1towername, b1names = name, ["conv"]
		b2towername, b2names = "%s_tower" % name, ["conv", "conv_1"]
		b3towername, b3names = "%s_tower_1" % name, ["conv", "conv_1", "conv_2"]
		b4towername, b4names = "%s_tower_2" % name, ["avg_pool", "conv"]

	else:
		raise ValueError("Unrecognized block type")

	branch1 = tower(
		b1towername, b1names, [inmaps] + b1m, [1], strides=[1], pads=[0], act=act, bn=bn, scheme=scheme, typ=typ
	)

	branch2 = tower(
		b2towername, b2names, [inmaps] + b2m, [1, b2size], strides=[1, 1], pads=[0, b2pad], act=act, bn=bn,
		scheme=scheme, typ=typ
	)

	branch3 = tower(
		b3towername, b3names, [inmaps] + b3m, [1, 3, 3], strides=[1, 1, 1], pads=[0, 1, 1], act=act, bn=bn,
		scheme=scheme, typ=typ
	)

	branch4 = tower(
		b4towername, b4names, [inmaps, inmaps] + b4m, [3, 1], strides=[1, 1], pads=[1, 0], act=act, bn=bn,
		scheme=scheme, typ=typ
	)

	block.append(Replicate(times=4))
	block.append(Parallel().append(branch1).append(branch2).append(branch3).append(branch4))
	block.append(Concat(axis=1, name="ch_concat_%s_chconcat" % name))

	return block


def bnShrinkBlock(inmaps, b1m, b2m, name, act, bn, scheme, b1deep=True, pad=1, typ="bn"):
	block = Sequential()

	if typ == "bn":
		if b1deep:
			b1towername, b1names = name, ["3x3_reduce","3x3"]
		else:
			b1towername, b1names = name, ["3x3"]

		b2towername, b2names = name, ["double_3x3_reduce", "double_3x3_0", "double_3x3_1"]
		b3towername, b3names = name, ["max_pool"]

	elif typ == "v3":
		if b1deep:
			b1towername, b1names = name, ["conv"]
		else:
			b1towername, b1names = name, ["conv"]

		b2towername, b2names = "%s_tower" % name, ["conv", "conv_1", "conv_2"]
		b3towername, b3names = name, ["max_pool"]

	else:
		raise ValueError("Unrecognized block type")

	if b1deep:
		branch1 = tower(
			b1towername, b1names, [inmaps] + b1m, [1, 3], [1, 2], [0, pad], act=act, bn=bn, scheme=scheme, typ=typ
		)
	else:
		branch1 = tower(
			b1towername, b1names, [inmaps] + b1m, [3], [2], [pad], act=act, bn=bn, scheme=scheme, typ=typ
		)

	branch2 = tower(
		b2towername, b2names, [inmaps] + b2m, [1, 3, 3], [1, 1, 2], [0, 1, pad], act=act, bn=bn, scheme=scheme, typ=typ
	)

	branch3 = tower(
		b3towername, b3names, [inmaps, inmaps], [3], [2], [pad], act=act, bn=bn, scheme=scheme, typ=typ
	)

	block.append(Replicate(times=3))
	block.append(Parallel().append(branch1).append(branch2).append(branch3))
	block.append(Concat(axis=1, name="ch_concat_%s_chconcat" % name))

	return block


def factorBlock(inmaps, b1m, b2m, b3m, b4m, name, act, bn, scheme):
	block = Sequential()

	b1towername, b1names = name, ["conv"]
	b2towername, b2names = "%s_tower" % name, ["conv", "conv_1", "conv_2"]
	b3towername, b3names = "%s_tower_1" % name, ["conv", "conv_1", "conv_2", "conv_3", "conv_4"]
	b4towername, b4names = "%s_tower_2" % name, ["avg_pool", "conv"]

	branch1 = tower(
		b1towername, b1names, [inmaps] + b1m, [1], [1], [0], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch2 = tower(
		b2towername, b2names, [inmaps] + b2m, [1, (1, 7), (7, 1)], [1, 1, 1], [0, (0, 3), (3, 0)], act=act, bn=bn,
		scheme=scheme, typ="v3"
	)

	branch3 = tower(
		b3towername, b3names, [inmaps] + b3m, [1, (7, 1), (1, 7), (7, 1), (1, 7)], [1, 1, 1, 1, 1],
		[0, (3, 0), (0, 3), (3, 0), (0, 3)], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch4 = tower(
		b4towername, b4names, [inmaps, inmaps] + b4m, [3, 1], [1, 1], [1, 0], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	block.append(Replicate(times=4))
	block.append(Parallel().append(branch1).append(branch2).append(branch3).append(branch4))
	block.append(Concat(axis=1, name="ch_concat_%s_chconcat" % name))

	return block


def v3ShrinkBlock(inmaps, b1m, b2m, name, act, bn, scheme):
	block = Sequential()

	b1towername, b1names = "%s_tower" % name, ["conv", "conv_1"]
	b2towername, b2names = "%s_tower_1" % name, ["conv", "conv_1", "conv_2", "conv_3"]
	b3towername, b3names = name, ["max_pool"]

	branch1 = tower(
		b1towername, b1names, [inmaps] + b1m, [1, 3], [1, 2], [0, 0], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch2 = tower(
		b2towername, b2names, [inmaps] + b2m, [1, (1, 7), (7, 1), 3], [1, 1, 1, 2], [0, (0, 3), (3, 0), 0],
		act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch3 = tower(b3towername, b3names, [inmaps, inmaps], [3], [2], [0], act=act, bn=bn, scheme=scheme, typ="v3")

	block.append(Replicate(times=3))
	block.append(Parallel().append(branch1).append(branch2).append(branch3))
	block.append(Concat(axis=1, name="ch_concat_%s_chconcat" % name))

	return block


def expandBlock(inmaps, b1m, b2m, b3m, b4m, name, act, bn, scheme, pool="avg"):
	block = Sequential()

	b1towername, b1names = name, ["conv"]
	b2towername, b2names, b2sub1names, b2sub2names = "%s_tower" % name, ["conv"], ["mixed_conv"], ["mixed_conv_1"]
	b3towername,b3names,b3sub1names,b3sub2names = "%s_tower_1"%name, ["conv","conv_1"], ["mixed_conv"], ["mixed_conv_1"]

	branch1 = tower(b1towername, b1names, [inmaps] + b1m, [1], [1], [0], act=act, bn=bn, scheme=scheme, typ="v3")

	branch2 = tower(b2towername, b2names, [inmaps, b2m[0]], [1], [1], [0], act=act, bn=bn, scheme=scheme, typ="v3")
	branch2sub1 = tower(
		b2towername, b2sub1names, [b2m[0], b2m[1]], [(1, 3)], [1], [(0, 1)], act=act, bn=bn, scheme=scheme, typ="v3"
	)
	branch2sub2 = tower(
		b2towername, b2sub2names, [b2m[0], b2m[2]], [(3, 1)], [1], [(1, 0)], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch2.append(Replicate(times=2))
	branch2.append(Parallel().append(branch2sub1).append(branch2sub2))

	branch3 = tower(
		b3towername, b3names, [inmaps, b3m[0], b3m[1]], [1, 3], [1, 1], [0, 1], act=act, bn=bn, scheme=scheme, typ="v3"
	)
	branch3sub1 = tower(
		b3towername, b3sub1names, [b3m[1], b3m[2]], [(1, 3)], [1], [(0, 1)], act=act, bn=bn, scheme=scheme, typ="v3"
	)
	branch3sub2 = tower(
		b3towername, b3sub2names, [b3m[1], b3m[3]], [(3, 1)], [1], [(1, 0)], act=act, bn=bn, scheme=scheme, typ="v3"
	)

	branch3.append(Replicate(times=2))
	branch3.append(Parallel().append(branch3sub1).append(branch3sub2))

	if pool == "avg":
		branch4 = tower(
			"%s_tower_2" % name, ["avg_pool", "conv"], [inmaps, inmaps] + b4m, [3, 1], [1, 1], [1, 0], act=act, bn=bn,
			scheme=scheme, typ="v3"
		)

	elif pool == "max":
		branch4 = tower(
			"%s_tower_2" % name, ["max_pool", "conv"], [inmaps, inmaps] + b4m, [3, 1], [1, 1], [1, 0], act=act, bn=bn,
			scheme=scheme, typ="v3"
		)

	else:
		raise ValueError("Unrecognized block type")

	block.append(Replicate(times=4))
	block.append(Parallel().append(branch1).append(branch2).append(branch3).append(branch4))
	block.append(ToList())
	block.append(Concat(axis=1, name="ch_concat_%s_chconcat" % name))

	return block


def unittest():
	bn = loadInceptionBN(None, initscheme="gaussian")

	data = gpuarray.to_gpu(np.random.randn(1, 3, 224, 224).astype(np.float32))
	bn(data)

	del bn
	gpuarray.memoryPool.freeHeld()

	v3 = loadInceptionV3(None, initscheme="gaussian")

	data = gpuarray.to_gpu(np.random.randn(1, 3, 299, 299).astype(np.float32))
	v3(data)

	del v3
	gpuarray.memoryPool.freeHeld()


if __name__ == "__main__":
	unittest()
