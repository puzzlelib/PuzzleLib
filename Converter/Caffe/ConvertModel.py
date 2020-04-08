import os, subprocess, pickle
from google.protobuf.descriptor import FieldDescriptor as FD

from PuzzleLib.Converter.Caffe import Parsers


def js2hdf(js, hdf, compress="gzip", netName=None, **kwargs):
	if not "layer" in js:
		Parsers.parseOldCaffeFormat(js, hdf, compress, netName)
	else:
		Parsers.parseNewCaffeFormat(js, hdf, compress, netName, **kwargs)


def pb2json(pb):
	ftype2js = {
		FD.TYPE_DOUBLE: float,
		FD.TYPE_FLOAT: float,
		FD.TYPE_INT64: int,
		FD.TYPE_UINT64: int,
		FD.TYPE_INT32: int,
		FD.TYPE_FIXED64: float,
		FD.TYPE_FIXED32: float,
		FD.TYPE_BOOL: bool,
		FD.TYPE_STRING: str,
		FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
		FD.TYPE_UINT32: int,
		FD.TYPE_ENUM: int,
		FD.TYPE_SFIXED32: float,
		FD.TYPE_SFIXED64: float,
		FD.TYPE_SINT32: int,
		FD.TYPE_SINT64: int,
	}

	js = {}
	fields = pb.ListFields()

	for field,value in fields:
		ftype = None

		if field.type == FD.TYPE_MESSAGE:
			ftype = pb2json
		elif field.type in ftype2js:
			ftype = ftype2js[field.type]
		else:
			print(
				"WARNING: Field %s.%s of type '%d' is not supported" %
				(pb.__class__.__name__, field.name, field.type)
			)

		js[field.name] = [ftype(v) for v in value] if field.label == FD.LABEL_REPEATED else ftype(value)

	return js


def isPickled(modelname):
	return os.path.isfile(os.path.splitext(modelname)[0] + ".pkl")


def savePickle(js, modelname):
	pickle.dump(js, open(os.path.splitext(modelname)[0] + ".pkl", "wb"))


def loadPickle(modelname):
	return pickle.load(open(os.path.splitext(modelname)[0] + ".pkl", "rb"))


def main():
	modelname = "VGG_ILSVRC_16_layers.caffemodel"
	netname = None
	batchNormVarInverse = False
	eps = 1e-5

	if not isPickled(modelname):
		print("Model is not pickled ...")

		subprocess.check_call(["protoc", "--proto_path", ".", "--python_out", ".", "caffe.proto"])
		print("Compiled caffe.proto")

		from PuzzleLib.Converter.Caffe import caffe_pb2
		net = caffe_pb2.NetParameter()

		msg = open(modelname, "rb").read()

		print("Started parsing caffemodel %s ... May take a lot of time ..." % modelname)
		net.ParseFromString(msg)

		print("Started jsoning caffemodel ...")
		js = pb2json(net)
		del net

		print("Saving pickle ...")
		savePickle(js, modelname)
	else:
		print("Loading pickle ...")
		js = loadPickle(modelname)

	print("Saving as hdf ...")

	js2hdf(
		js, os.path.splitext(modelname)[0] + ".hdf", compress="gzip", netName=netname,
		batchNormVarInverse=batchNormVarInverse, eps=eps
	)


if __name__ == "__main__":
	main()
