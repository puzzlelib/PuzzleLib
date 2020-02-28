import h5py
import numpy as np


def parseOldCaffeFormat(js, hdf, compress="gzip", netName=None):
	paramlayers = {4: "convolution", 39: "deconvolution", 14: "inner_product"}
	oldparamlayers = {"conv": "convolution", "innerproduct": "inner_product"}

	if isinstance(hdf, str):
		hdf = h5py.File(hdf, "w")

	linkGrp = hdf.create_group("links")
	paramGrp = hdf.create_group("params")
	hdf.require_group("attrs")

	layers = js["layers"]

	if netName is None:
		netName = js["name"]

	paramIdx = 0
	for layer in layers:
		if "layer" in layer:
			layer = layer["layer"]

		layerName = "%s.%s" % (netName, layer["name"])

		if layer["type"] in paramlayers:
			layertype = paramlayers[layer["type"]]

		elif layer["type"] in oldparamlayers:
			layertype = oldparamlayers[layer["type"]]

		else:
			continue

		if layertype == "convolution":
			blobs = layer["blobs"]
			for blob in blobs:
				param = np.array(blob["data"], dtype=np.float32)
				if blob["num"] == blob["channels"] == blob["height"] == 1:
					b = np.reshape(param, (1, param.shape[0], 1, 1))
					linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=b, compression=compress)

				else:
					W = np.reshape(param, (blob["num"], blob["channels"], blob["height"], blob["width"]))
					linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=W, compression=compress)

				paramIdx += 1

		elif layertype == "inner_product":
			blobs = layer["blobs"]
			for blob in blobs:
				param = np.array(blob["data"], dtype=np.float32)
				if blob["num"] == blob["channels"] == blob["height"] == 1:
					b = np.reshape(param, (param.shape[0], ))
					linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=b, compression=compress)

				else:
					W = np.reshape(param, (blob["height"], blob["width"])).T
					linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=W, compression=compress)

				paramIdx += 1

		else:
			raise NotImplementedError()


def parseNewCaffeFormat(js, hdf, compress="gzip", netName=None, **kwargs):
	paramlayers = {"Convolution", "Deconvolution", "InnerProduct", "BatchNorm", "Scale", "PReLU"}

	if isinstance(hdf, str):
		hdf = h5py.File(hdf, "w")

	linkGrp = hdf.create_group("links")
	paramGrp = hdf.create_group("params")
	attrGrp = hdf.require_group("attrs")

	layers = js["layer"]

	if netName is None:
		netName = js["name"]

	paramIdx = 0
	for i, layer in enumerate(layers):
		if layer["type"] in paramlayers:
			layertype = layer["type"]
			layerName = "%s.%s" % (netName, layer["name"])

			if layertype == "Convolution":
				blobs = layer["blobs"]
				for blob in blobs:
					param = np.array(blob["data"], dtype=np.float32)
					dim = blob["shape"]["dim"]
					if len(dim) == 1:
						b = np.reshape(param, (1, param.shape[0], 1, 1))
						linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
						paramGrp.create_dataset(str(paramIdx), data=b, compression=compress)

					else:
						W = np.reshape(param, dim)
						linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
						paramGrp.create_dataset(str(paramIdx), data=W, compression=compress)

					paramIdx += 1

			elif layertype == "InnerProduct":
				blobs = layer["blobs"]
				for blob in blobs:
					param = np.array(blob["data"], dtype=np.float32)
					dim = blob["shape"]["dim"]
					if len(dim) == 1:
						b = np.reshape(param, (param.shape[0], ))
						linkGrp.create_dataset("%s.b" % layerName, data=paramIdx)
						paramGrp.create_dataset(str(paramIdx), data=b, compression=compress)

					else:
						W = np.reshape(param, dim).T
						linkGrp.create_dataset("%s.W" % layerName, data=paramIdx)
						paramGrp.create_dataset(str(paramIdx), data=W, compression=compress)

					paramIdx += 1

			elif layertype == "BatchNorm":
				blobs = layer["blobs"]
				dim = blobs[0]["shape"]["dim"][0]

				mean = np.array(blobs[0]["data"], dtype=np.float32).reshape((1, dim, 1, 1))
				var = np.array(blobs[1]["data"], dtype=np.float32).reshape((1, dim, 1, 1))

				if len(blobs) > 2:
					scale = blobs[2]["data"][0]

					if scale > 0.0:
						scale = 1.0 / scale

					mean *= scale
					var *= scale

				if "batchNormVarInverse" in kwargs and kwargs["batchNormVarInverse"]:
					var = 1 / np.sqrt(var + kwargs["eps"])

				attrGrp.create_dataset("%s.mean" % layerName, data=mean)
				attrGrp.create_dataset("%s.var" % layerName, data=var)

			elif layertype == "Scale":
				if i > 0 and layers[i-1]["type"] == "BatchNorm":
					blobs = layer["blobs"]
					dim = blobs[0]["shape"]["dim"][0]

					lastLayerName = "%s.%s" % (netName, layers[i - 1]["name"])

					scale = np.array(blobs[0]["data"], dtype=np.float32).reshape((1, dim, 1, 1))
					linkGrp.create_dataset("%s.scale" % lastLayerName, data=paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=scale, compression=compress)
					paramIdx += 1

					if len(blobs) > 1:
						bias = np.array(blobs[1]["data"], dtype=np.float32).reshape((1, dim, 1, 1))
						linkGrp.create_dataset("%s.bias" % lastLayerName, data=paramIdx)
						paramGrp.create_dataset(str(paramIdx), data=bias, compression=compress)
						paramIdx += 1

			elif layertype == "PReLU":
				blobs = layer["blobs"]
				if len(blobs) > 0:
					blob = blobs[0]
					slopes = np.array(blob["data"], dtype=np.float32)
					linkGrp.create_dataset("%s.slopes" % layerName, data = paramIdx)
					paramGrp.create_dataset(str(paramIdx), data=slopes, compression=compress)
					paramIdx += 1
