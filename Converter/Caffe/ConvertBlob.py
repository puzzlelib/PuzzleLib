import subprocess

import numpy as np
import h5py


def saveAttr(data, name, filename):
	hdf = h5py.File(filename, mode="a")

	modelname = next(iter(hdf["links"].keys())).split(sep=".")[0]

	attrGrpName = "attrs.%s" % modelname

	attrGrp = hdf.require_group(attrGrpName)
	attrGrp.create_dataset("%s.%s" % (modelname, name), data=data)


def main():
	binaryname = "ResNet_mean.binaryproto"
	modelname = "ResNet-50-model.hdf"
	attrName = "mean"

	subprocess.check_call(["protoc", "--proto_path", ".", "--python_out", ".", "caffe.proto"])
	print("Compiled caffe.proto")

	from PuzzleLib.Converter.Caffe import caffe_pb2
	blob = caffe_pb2.BlobProto()

	msg = open(binaryname, "rb").read()

	print("Started parsing binaryproto %s ..." % (binaryname))
	blob.ParseFromString(msg)

	data = np.array(blob.data, dtype=np.float32).reshape((1, blob.channels, blob.height, blob.width))
	saveAttr(data, attrName, modelname)


if __name__ == "__main__":
	main()
