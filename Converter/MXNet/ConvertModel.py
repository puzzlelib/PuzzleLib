import struct, array, enum, os, json

import numpy as np
import h5py


class TypeFlag(enum.Enum):
	kFloat32 = 0
	kFloat64 = 1
	kFloat16 = 2
	kUint8 = 3
	kInt32 = 4


def readHeader(file):
	magic, reserved = struct.unpack("<QQ", file.read(16))

	if magic != 0x112:
		raise ValueError()


def readData(file):
	tensors = []
	ntensors = struct.unpack("<Q", file.read(8))[0]

	for i in range(ntensors):
		ndim = struct.unpack("<I", file.read(4))[0]
		shape = struct.unpack("<" + "I" * ndim, file.read(4 * ndim))

		devtype, devid, typeflag = struct.unpack("<iii", file.read(12))
		typeflag = TypeFlag(typeflag)

		if typeflag == TypeFlag.kFloat32:
			ltrl = "f"
		elif typeflag == TypeFlag.kFloat64:
			ltrl = "d"
		elif typeflag == TypeFlag.kFloat16:
			ltrl = "h"
		elif typeflag == TypeFlag.kUint8:
			ltrl = "B"
		elif typeflag == TypeFlag.kInt32:
			ltrl = "i"
		else:
			raise ValueError()

		data = array.array(ltrl)
		data.fromfile(file, int(np.prod(shape)))

		tensor = np.array(data).reshape(shape)
		tensors.append(tensor)

	return tensors


def readKeys(file):
	keys = []
	nkeys = struct.unpack("<Q", file.read(8))[0]

	for i in range(nkeys):
		len = struct.unpack("<Q", file.read(8))[0]
		data = array.array("B")
		data.fromfile(file, len)

		key = data.tobytes().decode()
		keys.append(key)

	return keys


def loadSymbols(symbolsname):
	with open(symbolsname) as file:
		symbols = json.loads(file.read())
		return symbols


def buildHdf(keys, tensors, symbols, hdf, modelname, compress="gzip"):
	hdf = h5py.File(hdf, "w") if isinstance(hdf, str) else hdf

	table = {}
	for i in range(len(keys)):
		table[keys[i]] = tensors[i]

	linkGrp = hdf.create_group("links")
	paramGrp = hdf.create_group("params")
	attrGrp = hdf.create_group("attrs")

	paramIdx = 0

	for i in range(len(symbols["nodes"])):
		node = symbols["nodes"][i]

		name = node["name"]
		layerName = "%s.%s" % (modelname, name)

		op = node["op"]

		if op == "Convolution":
			if ("arg:%s_weight" % name) in keys:
				linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
				paramGrp.create_dataset(str(paramIdx), data=table["arg:%s_weight" % name], compression=compress)
				paramIdx += 1

			if ("arg:%s_bias" % name) in keys:
				linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
				bias = table["arg:%s_bias" % name]
				paramGrp.create_dataset(str(paramIdx), data=bias.reshape(1, bias.shape[0], 1, 1), compression=compress)
				paramIdx += 1

		elif op == "BatchNorm":
			if ("arg:%s_gamma" % name) in keys:
				linkGrp.create_dataset("%s.scale" % layerName, data=paramIdx)
				scale = table["arg:%s_gamma" % name]
				paramGrp.create_dataset(str(paramIdx), data=scale.reshape(1, scale.shape[0], 1,1), compression=compress)
				paramIdx += 1

			if ("arg:%s_beta" % name) in keys:
				linkGrp.create_dataset("%s.bias" % layerName, data=paramIdx)
				bias = table["arg:%s_beta" % name]
				paramGrp.create_dataset(str(paramIdx), data=bias.reshape(1, bias.shape[0], 1, 1), compression=compress)
				paramIdx += 1

			if ("aux:%s_moving_mean" % name) in keys:
				mean = table["aux:%s_moving_mean" % name]
				attrGrp.create_dataset("%s.mean" % layerName, data=mean.reshape(1, mean.shape[0], 1, 1))

			if ("aux:%s_moving_var" % name) in keys:
				var = table["aux:%s_moving_var" % name]
				attrGrp.create_dataset("%s.var" % layerName, data=var.reshape(1, var.shape[0], 1, 1))

		elif op == "FullyConnected":
			if ("arg:%s_weight" % name) in keys:
				linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
				paramGrp.create_dataset(str(paramIdx), data=table["arg:%s_weight" % name].T, compression=compress)
				paramIdx += 1

			if ("arg:%s_bias" % name) in keys:
				linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
				paramGrp.create_dataset(str(paramIdx), data=table["arg:%s_bias" % name], compression=compress)
				paramIdx += 1


def main():
	modelname = "Inception-7-0001.params"
	symbolsname = "Inception-7-symbol.json"

	print("Deserializing mxnet model ...")
	with open(modelname, mode="rb") as file:
		readHeader(file)
		tensors = readData(file)
		keys = readKeys(file)

	symbols = loadSymbols(symbolsname)

	print("Parsing tensor dictionary and saving hdf ...")
	name = os.path.basename(os.path.splitext(modelname)[0])
	buildHdf(keys, tensors, symbols, os.path.splitext(modelname)[0] + ".hdf", name)


if __name__ == "__main__":
	main()
