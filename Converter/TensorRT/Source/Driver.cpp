#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#ifdef __GNUC__
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <NvOnnxParser.h>

#ifdef __GNUC__
	#pragma GCC diagnostic pop
#endif

#include "Plugins.h"


enum class DataType : int
{
	float_ = static_cast<int>(nv::DataType::kFLOAT),
	int8 = static_cast<int>(nv::DataType::kINT8),
	half = static_cast<int>(nv::DataType::kHALF)
};


enum class ActivationType : int
{
	relu = static_cast<int>(nv::ActivationType::kRELU),
	leakyRelu = static_cast<int>(nv::ActivationType::kLEAKY_RELU),
	clip = static_cast<int>(nv::ActivationType::kCLIP),
	sigmoid = static_cast<int>(nv::ActivationType::kSIGMOID),
	tanh = static_cast<int>(nv::ActivationType::kTANH)
};


enum class RNNMode : int
{
	relu = static_cast<int>(nv::RNNOperation::kRELU),
	tanh = static_cast<int>(nv::RNNOperation::kTANH),
	lstm = static_cast<int>(nv::RNNOperation::kLSTM),
	gru = static_cast<int>(nv::RNNOperation::kGRU)
};


enum class RNNDirection : int
{
	uni = static_cast<int>(nv::RNNDirection::kUNIDIRECTION),
	bi = static_cast<int>(nv::RNNDirection::kBIDIRECTION)
};


enum class RNNInputMode : int
{
	linear = static_cast<int>(nv::RNNInputMode::kLINEAR),
	skip = static_cast<int>(nv::RNNInputMode::kSKIP)
};


enum class RNNGateType : int
{
	input = static_cast<int>(nv::RNNGateType::kINPUT),
	output = static_cast<int>(nv::RNNGateType::kOUTPUT),
	forget = static_cast<int>(nv::RNNGateType::kFORGET),
	update = static_cast<int>(nv::RNNGateType::kUPDATE),
	reset = static_cast<int>(nv::RNNGateType::kRESET),
	cell = static_cast<int>(nv::RNNGateType::kCELL),
	hidden = static_cast<int>(nv::RNNGateType::kHIDDEN)
};


struct Tensor
{
	nv::ITensor *m_tensor;


	void setName(const char *name)
	{
		m_tensor->setName(name);
	}

	std::string getName()
	{
		return m_tensor->getName();
	}

	std::vector<int> getShape()
	{
		auto dims = m_tensor->getDimensions();
		return std::vector<int>(dims.d, dims.d + dims.nbDims);
	}
};


struct ConsoleLogger : nv::ILogger
{
	bool m_enabled;


	ConsoleLogger(bool enabled)
	{
		m_enabled = enabled;
	}


	void log(nv::ILogger::Severity severity, const char *msg) override
	{
		if (!m_enabled)
			return;

		switch (severity)
		{
			case Severity::kINTERNAL_ERROR:
			{
				std::cerr << "[TensorRT] INTERNAL_ERROR: ";
				break;
			}
			case Severity::kERROR:
			{
				 std::cerr << "[TensorRT] ERROR: ";
				 break;
			}
			case Severity::kWARNING:
			{
				std::cerr << "[TensorRT] WARNING: ";
				break;
			}
			case Severity::kINFO:
			{
				std::cout << "[TensorRT] INFO: " << msg << std::endl;
				return;
			}
			default:
			{
				std::cerr << "[TensorRT] UNKNOWN: ";
				break;
			}
		}
		std::cerr << msg << std::endl;
	}
};


struct ICalibrator : nv::IInt8EntropyCalibrator2
{
	ICalibrator() = default;

	int getBatchSize() const override
	{
		PYBIND11_OVERLOAD_PURE_NAME(int, nv::IInt8EntropyCalibrator2, "getBatchSize", getBatchSize);
	}

	bool getBatch(void *bindings[], const char *names[], int nbBindings) override
	{
		py::list pybindings, pynames;
		for (int i = 0; i < nbBindings; i++)
		{
			pybindings.append(reinterpret_cast<std::size_t>(&bindings[i]));
			pynames.append(names[i]);
		}

		PYBIND11_OVERLOAD_PURE_NAME(bool, nv::IInt8EntropyCalibrator2, "getBatch", getBatch, pybindings, pynames);
	}

	const void *readCalibrationCache(std::size_t& length) override
	{
		length = 0;
		return nullptr;
	}

	void writeCalibrationCache(const void *, std::size_t) override
	{

	}
};


void buildRTEngine(nvinfer1::INetworkDefinition *network, nv::IBuilder *builder, int batchsize, int workspace,
				   DataType mode, ICalibrator *calibrator, std::string savepath)
{
	builder->setMaxBatchSize(batchsize);
	builder->setMaxWorkspaceSize(workspace);

	if (mode == DataType::int8)
	{
		builder->setInt8Mode(true);
		builder->setInt8Calibrator(calibrator);
	}
	else if (mode == DataType::half)
	{
		builder->setFp16Mode(true);
	}

	nv::ICudaEngine *engine = builder->buildCudaEngine(*network);
	if (engine == nullptr)
		throw std::runtime_error("Failed to create engine");

	auto stream = engine->serialize();

	std::ofstream file(savepath, std::ios::binary);
	if (!file.is_open())
		throw std::invalid_argument("Invalid engine save path: " + savepath);

	file.write(reinterpret_cast<char *>(stream->data()), stream->size());
	stream->destroy();

	engine->destroy();
}


void buildRTEngineFromCaffe(std::string prototxt, std::string caffemodel, int batchsize, py::list outlayers,
							DataType mode, ICalibrator *calibrator, int workspace, std::string savepath, bool log)
{
	ConsoleLogger logger(log);
	nv::IBuilder *builder = nv::createInferBuilder(logger);

	if (mode == DataType::int8 && !builder->platformHasFastInt8())
		throw std::invalid_argument("INT8 datatype is not supported on this platform");

	else if (mode == DataType::half && !builder->platformHasFastFp16())
		throw std::invalid_argument("FP16 datatype is not supported on this platform");

	nv::INetworkDefinition *network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();

	CaffePluginFactory factory;
	parser->setPluginFactory(&factory);

	auto blobNameToTensor = parser->parse(
		prototxt.c_str(), caffemodel.c_str(), *network, static_cast<nv::DataType>(mode)
	);

	for (std::size_t i = 0; i < len(outlayers); i++)
	{
		std::string outlayer = py::cast<std::string>(outlayers[i]);
		network->markOutput(*blobNameToTensor->find(outlayer.c_str()));
	}

	buildRTEngine(network, builder, batchsize, workspace, mode, calibrator, savepath);

	parser->destroy();

	network->destroy();
	builder->destroy();
}


void buildRTEngineFromOnnx(std::string onnxname, int batchsize, DataType mode, ICalibrator *calibrator, int workspace,
						   std::string savepath, bool log)
{
	ConsoleLogger logger(log);
	nv::IBuilder *builder = nv::createInferBuilder(logger);

	if (mode == DataType::int8 && !builder->platformHasFastInt8())
		throw std::invalid_argument("INT8 datatype is not supported on this platform");

	else if (mode == DataType::half && !builder->platformHasFastFp16())
		throw std::invalid_argument("FP16 datatype is not supported on this platform");

	nvinfer1::INetworkDefinition *network = builder->createNetwork();
	nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);

	if (!parser->parseFromFile(onnxname.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
	{
		const char *msg = "failed to parse onnx file";
		logger.log(nvinfer1::ILogger::Severity::kERROR, msg);

		throw std::invalid_argument(msg);
	}

	buildRTEngine(network, builder, batchsize, workspace, mode, calibrator, savepath);

	parser->destroy();

	network->destroy();
	builder->destroy();
}


struct Graph
{
	ConsoleLogger m_logger;

	nv::IBuilder *m_builder;
	nv::INetworkDefinition *m_graph;

	PluginFactory m_pluginFactory;


	Graph(bool log) : m_logger(log)
	{
		m_builder = nv::createInferBuilder(m_logger);
		m_graph = m_builder->createNetwork();
	}

	~Graph()
	{
		if (m_graph != nullptr) m_graph->destroy();
		if (m_builder != nullptr) m_builder->destroy();
	}

	bool platformHasFastFp16()
	{
		return m_builder->platformHasFastFp16();
	}

	void setFp16Mode(bool mode)
	{
		m_builder->setFp16Mode(mode);
	}

	bool platformHasFastInt8()
	{
		return m_builder->platformHasFastInt8();
	}

	void setInt8Mode(bool mode)
	{
		m_builder->setInt8Mode(mode);
	}

	void setInt8Calibrator(ICalibrator *calibrator)
	{
		m_builder->setInt8Calibrator(calibrator);
	}

	void markOutput(Tensor tensor)
	{
		m_graph->markOutput(*tensor.m_tensor);
	}

	void setMaxBatchSize(int batchsize)
	{
		m_builder->setMaxBatchSize(batchsize);
	}

	void setMaxWorkspaceSize(std::size_t size)
	{
		m_builder->setMaxWorkspaceSize(size);
	}

	void buildCudaEngine(const char *savepath)
	{
		nv::ICudaEngine *engine = m_builder->buildCudaEngine(*m_graph);
		if (engine == nullptr)
			throw std::runtime_error("Failed to create engine");

		auto stream = engine->serialize();

		std::ofstream file(savepath, std::ios::binary);
		if (!file.is_open())
			throw std::invalid_argument("Invalid engine save path: " + std::string(savepath));

		file.write(reinterpret_cast<char *>(stream->data()), stream->size());

		stream->destroy();
		engine->destroy();
	}

	Tensor addInput(const char *name, DataType dtype, py::tuple shape)
	{
		nv::Dims dims;
		dims.nbDims = static_cast<int>(py::len(shape));

		for (int i = 0; i < dims.nbDims; i++)
		{
			dims.type[i] = (i == 0) ? nv::DimensionType::kCHANNEL : nv::DimensionType::kSPATIAL;
			dims.d[i] = py::cast<int>(shape[i]);
		}

		Tensor tensor = {m_graph->addInput(name, static_cast<nv::DataType>(dtype), dims)};
		return tensor;
	}

	Tensor addConvolution(Tensor input, int outmaps, py::tuple kernel, std::size_t Wdata, int64_t Wlen,
						  std::size_t biasdata, int64_t biaslen, py::tuple stride, py::tuple pad, py::tuple pydilation,
						  bool isDeconvolution, const char *name)
	{
		auto kernelSize = nv::DimsHW(py::cast<int>(kernel[0]), py::cast<int>(kernel[1]));
		nv::Weights W = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata), Wlen};

		nv::Weights bias;
		bias.type = nv::DataType::kFLOAT;

		if (biasdata != 0)
		{
			bias.values = reinterpret_cast<void *>(biasdata);
			bias.count = biaslen;
		}
		else
		{
			bias.values = nullptr;
			bias.count = 0;
		}

		auto striding = nv::DimsHW(py::cast<int>(stride[0]), py::cast<int>(stride[1]));
		auto padding = nv::DimsHW(py::cast<int>(pad[0]), py::cast<int>(pad[1]));

		nv::ILayer *layer = nullptr;
		if (isDeconvolution)
		{
			auto deconv = m_graph->addDeconvolution(*input.m_tensor, outmaps, kernelSize, W, bias);
			deconv->setName(name);

			deconv->setStride(striding);
			deconv->setPadding(padding);

			layer = deconv;
		}
		else
		{
			auto conv = m_graph->addConvolution(*input.m_tensor, outmaps, kernelSize, W, bias);
			conv->setName(name);

			conv->setStride(striding);
			conv->setPadding(padding);

			auto dilation = nv::DimsHW(py::cast<int>(pydilation[0]), py::cast<int>(pydilation[1]));
			conv->setDilation(dilation);

			layer = conv;
		}

		Tensor tensor = {layer->getOutput(0)};
		return tensor;
	}

	Tensor addScale(Tensor input, std::size_t shiftdata, std::size_t scaledata, std::size_t powerdata, int64_t len,
					const char *name)
	{
		nv::Weights shift = {nv::DataType::kFLOAT, reinterpret_cast<void *>(shiftdata), len};
		nv::Weights scaling = {nv::DataType::kFLOAT, reinterpret_cast<void *>(scaledata), len};
		nv::Weights power = {nv::DataType::kFLOAT, reinterpret_cast<void *>(powerdata), len};

		auto scale = m_graph->addScale(*input.m_tensor, nv::ScaleMode::kCHANNEL, shift, scaling, power);
		scale->setName(name);

		Tensor tensor = {scale->getOutput(0)};
		return tensor;
	}

	Tensor addActivation(Tensor input, ActivationType type, float alpha, float beta, const char *name)
	{
		auto act = m_graph->addActivation(*input.m_tensor, static_cast<nv::ActivationType>(type));

		act->setAlpha(alpha);
		act->setBeta(beta);
		act->setName(name);

		Tensor tensor = {act->getOutput(0)};
		return tensor;
	}

	Tensor addPooling(Tensor input, bool avg, py::tuple kernel, py::tuple stride, py::tuple pad, const char *name)
	{
		auto kernelSize = nv::DimsHW(py::cast<int>(kernel[0]), py::cast<int>(kernel[1]));

		auto pool = m_graph->addPooling(
			*input.m_tensor, avg ? nv::PoolingType::kAVERAGE : nv::PoolingType::kMAX, kernelSize
		);
		pool->setName(name);

		auto striding = nv::DimsHW(py::cast<int>(stride[0]), py::cast<int>(stride[1]));
		pool->setStride(striding);

		auto padding = nv::DimsHW(py::cast<int>(pad[0]), py::cast<int>(pad[1]));
		pool->setPadding(padding);

		Tensor tensor = {pool->getOutput(0)};
		return tensor;
	}

	Tensor addCrossMapLRN(Tensor input, int N, float alpha, float beta, float K, const char *name)
	{
		auto lrn = m_graph->addLRN(*input.m_tensor, N, alpha, beta, K);
		lrn->setName(name);

		Tensor tensor = {lrn->getOutput(0)};
		return tensor;
	}

	Tensor addAdd(Tensor input1, Tensor input2, const char *name)
	{
		auto add = m_graph->addElementWise(*input1.m_tensor, *input2.m_tensor, nv::ElementWiseOperation::kSUM);
		add->setName(name);

		Tensor tensor = {add->getOutput(0)};
		return tensor;
	}

	Tensor addConcatenation(py::list inputs, const char *name)
	{
		std::vector<nv::ITensor *> tensors(py::len(inputs));

		for (std::size_t i = 0; i < py::len(inputs); i++)
		{
			Tensor tensor = py::cast<Tensor>(inputs[i]);
			tensors[i] = tensor.m_tensor;
		}

		auto concat = m_graph->addConcatenation(&tensors[0], static_cast<int>(tensors.size()));
		concat->setName(name);

		Tensor tensor = {concat->getOutput(0)};
		return tensor;
	}

	Tensor addFlatten(Tensor input, const char *name)
	{
		auto flatten = m_graph->addShuffle(*input.m_tensor);
		flatten->setName(name);

		auto indims = input.m_tensor->getDimensions();

		nv::Dims outdims = {};
		outdims.nbDims = 1;
		outdims.d[0] = 1;

		for (int i = 0; i < indims.nbDims; i++)
			outdims.d[0] *= indims.d[i];

		flatten->setReshapeDimensions(outdims);

		Tensor tensor = {flatten->getOutput(0)};
		return tensor;
	}

	Tensor addLinear(Tensor input, int outputs, std::size_t Wdata, int64_t Wlen,
					 std::size_t biasdata, int64_t biaslen, const char *name)
	{
		nv::Weights W = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata), Wlen};

		nv::Weights bias;
		bias.type = nv::DataType::kFLOAT;

		if (biasdata != 0)
		{
			bias.values = reinterpret_cast<void *>(biasdata);
			bias.count = biaslen;
		}
		else
		{
			bias.values = nullptr;
			bias.count = 0;
		}

		auto indims = input.m_tensor->getDimensions();
		auto inReshape = m_graph->addShuffle(*input.m_tensor);

		nv::Dims inReshapeDims;
		inReshapeDims.nbDims = 3;
		inReshapeDims.d[0] = indims.d[0];
		inReshapeDims.d[1] = inReshapeDims.d[2] = 1;

		inReshape->setReshapeDimensions(inReshapeDims);

		auto linear = m_graph->addFullyConnected(*inReshape->getOutput(0), outputs, W, bias);
		linear->setName(name);

		auto outReshape = m_graph->addShuffle(*linear->getOutput(0));

		nv::Dims outReshapeDims;
		outReshapeDims.nbDims = 1;
		outReshapeDims.d[0] = outputs;

		outReshape->setReshapeDimensions(outReshapeDims);

		Tensor tensor = {outReshape->getOutput(0)};
		return tensor;
	}

	Tensor addSoftMax(Tensor input, const char *name)
	{
		auto softmax = m_graph->addSoftMax(*input.m_tensor);
		softmax->setName(name);

		Tensor tensor = {softmax->getOutput(0)};
		return tensor;
	}

	Tensor addSwapAxes(Tensor input, int axis1, int axis2, const char *name)
	{
		auto swapaxes = m_graph->addShuffle(*input.m_tensor);
		swapaxes->setName(name);

		nv::Permutation permutation;
		for (int i = 0; i < nv::Dims::MAX_DIMS; i++)
			permutation.order[i] = i;

		permutation.order[axis1] = axis2;
		permutation.order[axis2] = axis1;

		swapaxes->setFirstTranspose(permutation);

		Tensor tensor = {swapaxes->getOutput(0)};
		return tensor;
	}

	Tensor addMoveAxis(Tensor input, int src, int dst, const char *name)
	{
		auto moveaxis = m_graph->addShuffle(*input.m_tensor);
		moveaxis->setName(name);

		nv::Permutation permutation;
		for (int i = 0; i < nv::Dims::MAX_DIMS; i++)
		{
			int order = 0;

			if ((i < src && i < dst) || (i > src && i > dst))
				order = i;

			else if (i == dst)
				order = src;

			else
				order = i + (src < dst);

			permutation.order[i] = order;
		}

		moveaxis->setFirstTranspose(permutation);

		Tensor tensor = {moveaxis->getOutput(0)};
		return tensor;
	}

	std::vector<Tensor> addSplit(Tensor input, int axis, std::vector<int> sections, const char *name)
	{
		assert(axis == 1);
		std::vector<Tensor> tensors(sections.size());

		auto inshape = input.getShape();
		int offset = 0;

		for (std::size_t i = 0; i < sections.size(); i++)
		{
			nv::Dims start, size, stride;

			start.nbDims = static_cast<int>(inshape.size());
			start.d[0] = offset;
			offset += sections[i];

			for (std::size_t d = 0; d < inshape.size() - 1; d++)
				start.d[d + 1] = 0;

			size.nbDims = static_cast<int>(inshape.size());
			size.d[0] = sections[i];
			for (std::size_t d = 0; d < inshape.size() - 1; d++)
				size.d[1 + d] = inshape[1 + d];

			stride.nbDims = static_cast<int>(inshape.size());
			for (std::size_t d = 0; d < inshape.size(); d++)
				stride.d[d] = 1;

			auto slice = m_graph->addSlice(*input.m_tensor, start, size, stride);

			auto sliceName = std::string(name) + "_slice_" + std::to_string(i);
			slice->setName(sliceName.c_str());

			Tensor tensor = {slice->getOutput(0)};
			tensors[i] = tensor;
		}

		return tensors;
	}

	Tensor addReshape(Tensor input, py::tuple shape, const char *name)
	{
		auto reshape = m_graph->addShuffle(*input.m_tensor);
		reshape->setName(name);

		nv::Dims dims;
		dims.nbDims = static_cast<int>(py::len(shape));

		for (int i = 0; i < dims.nbDims; i++)
		{
			dims.type[i] = (i == 0) ? nv::DimensionType::kCHANNEL : nv::DimensionType::kSPATIAL;
			dims.d[i] = py::cast<int>(shape[i]);
		}

		reshape->setReshapeDimensions(dims);

		Tensor tensor = {reshape->getOutput(0)};
		return tensor;
	}

	Tensor addGroupLinear(Tensor input, int groups, int insize, int outsize, std::size_t Wdata, int64_t Wlen,
						  std::size_t biasdata, int64_t biaslen, const char *name)
	{
		nv::Dims dims;
		dims.nbDims = 2;

		dims.d[0] = insize;
		dims.d[1] = outsize;

		nv::Weights W = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata), Wlen};
		auto weights = m_graph->addConstant(dims, W);

		nv::ILayer *linear = m_graph->addMatrixMultiply(*input.m_tensor, false, *weights->getOutput(0), false);
		linear->setName(name);

		if (biasdata != 0)
		{
			nv::Dims biasDims;
			biasDims.nbDims = 2;

			biasDims.d[0] = groups;
			biasDims.d[1] = outsize;

			nv::Weights b = {nv::DataType::kFLOAT, reinterpret_cast<void *>(biasdata), biaslen};

			auto bias = m_graph->addConstant(biasDims, b);
			linear = m_graph->addElementWise(*linear->getOutput(0), *bias->getOutput(0),
											 nv::ElementWiseOperation::kSUM);
		}

		Tensor tensor = {linear->getOutput(0)};
		return tensor;
	}

	Tensor addSum(Tensor input, int axis, const char *name)
	{
		auto sum = m_graph->addReduce(*input.m_tensor, nv::ReduceOperation::kSUM, axis, false);
		sum->setName(name);

		Tensor tensor = {sum->getOutput(0)};
		return tensor;
	}

	void updateRNNParams(nv::IRNNv2Layer *rnn, int layers,
						 const std::vector<std::size_t>& Wdata, const std::vector<int64_t>& Wlen,
						 const std::vector<std::size_t>& biasdata, const std::vector<int64_t>& biaslen)
	{
		for (int layer = 0; layer < layers; layer++)
		{
			int keys = 2;
			for (int k = 0; k < keys; k++)
			{
				int idx = keys * layer + k;
				nv::Weights weights;

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata[idx]), Wlen[idx]};
				rnn->setWeightsForGate(layer, nv::RNNGateType::kINPUT, k < keys / 2, weights);

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(biasdata[idx]), biaslen[idx]};
				rnn->setBiasForGate(layer, nv::RNNGateType::kINPUT, k < keys / 2, weights);
			}
		}
	}

	void updateLSTMParams(nv::IRNNv2Layer *rnn, int layers,
						  const std::vector<std::size_t>& Wdata, const std::vector<int64_t>& Wlen,
						  const std::vector<std::size_t>& biasdata, const std::vector<int64_t>& biaslen)
	{
		for (int layer = 0; layer < layers; layer++)
		{
			int keys = 8;
			for (int k = 0; k < keys; k++)
			{
				int idx = keys * layer + k;
				nv::Weights weights;

				nv::RNNGateType gate = nv::RNNGateType::kFORGET;
				switch (k)
				{
					case 0: case 4: gate = nv::RNNGateType::kFORGET; break;
					case 1: case 5: gate = nv::RNNGateType::kINPUT;  break;
					case 2: case 6: gate = nv::RNNGateType::kCELL;   break;
					case 3: case 7: gate = nv::RNNGateType::kOUTPUT; break;
					default: assert(false);
				}

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata[idx]), Wlen[idx]};
				rnn->setWeightsForGate(layer, gate, k < keys / 2, weights);

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(biasdata[idx]), biaslen[idx]};
				rnn->setBiasForGate(layer, gate, k < keys / 2, weights);
			}
		}
	}

	void updateGRUParams(nv::IRNNv2Layer *rnn, int layers,
						 const std::vector<std::size_t>& Wdata, const std::vector<int64_t>& Wlen,
						 const std::vector<std::size_t>& biasdata, const std::vector<int64_t>& biaslen)
	{
		for (int layer = 0; layer < layers; layer++)
		{
			int keys = 6;
			for (int k = 0; k < keys; k++)
			{
				int idx = keys * layer + k;
				nv::Weights weights;

				nv::RNNGateType gate = nv::RNNGateType::kUPDATE;
				switch (k)
				{
					case 0: case 3: gate = nv::RNNGateType::kUPDATE; break;
					case 1: case 4: gate = nv::RNNGateType::kRESET;  break;
					case 2: case 5: gate = nv::RNNGateType::kHIDDEN; break;
					default: assert(false);
				}

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata[idx]), Wlen[idx]};
				rnn->setWeightsForGate(layer, gate, k < keys / 2, weights);

				weights = {nv::DataType::kFLOAT, reinterpret_cast<void *>(biasdata[idx]), biaslen[idx]};
				rnn->setBiasForGate(layer, gate, k < keys / 2, weights);
			}
		}
	}

	Tensor addRNN(Tensor input, int layers, int hsize, int seqlen, RNNMode mode, RNNDirection direction,
				  RNNInputMode inputMode, std::vector<std::size_t> Wdata, std::vector<int64_t> Wlen,
				  std::vector<std::size_t> biasdata, std::vector<int64_t> biaslen, const char *name)
	{
		auto rnn = m_graph->addRNNv2(*input.m_tensor, layers, hsize, seqlen, static_cast<nv::RNNOperation>(mode));
		rnn->setName(name);

		rnn->setDirection(static_cast<nv::RNNDirection>(direction));
		rnn->setInputMode(static_cast<nv::RNNInputMode>(inputMode));

		int numDir = (direction == RNNDirection::uni) ? 1 : 2;

		switch (mode)
		{
			case RNNMode::relu:
			case RNNMode::tanh:
				updateRNNParams(rnn, layers * numDir, Wdata, Wlen, biasdata, biaslen);
				break;

			case RNNMode::lstm:
				updateLSTMParams(rnn, layers * numDir, Wdata, Wlen, biasdata, biaslen);
				break;

			case RNNMode::gru:
				updateGRUParams(rnn, layers * numDir, Wdata, Wlen, biasdata, biaslen);
				break;

			default:
			assert(false);
		}

		Tensor tensor = {rnn->getOutput(0)};
		return tensor;
	}

	Tensor addUpsample(Tensor input, int scale, const char *name)
	{
		auto upsample = m_graph->addResize(*input.m_tensor);
		upsample->setName(name);

		auto dims = input.m_tensor->getDimensions();

		std::vector<float> scales(dims.nbDims, static_cast<float>(scale));
		scales[0] = 1.0f;

		upsample->setScales(&scales[0], dims.nbDims);

		Tensor tensor = {upsample->getOutput(0)};
		return tensor;
	}

	Tensor addPRelu(Tensor input, std::size_t slopedata, int64_t slopelen, const char *name)
	{
		auto plugin = m_pluginFactory.createPRelu(reinterpret_cast<float *>(slopedata), slopelen);
		auto prelu = m_graph->addPluginExt(&input.m_tensor, 1, *plugin);

		auto internalName = name + PluginFactory::prelu;
		prelu->setName(internalName.c_str());

		Tensor tensor = {prelu->getOutput(0)};
		return tensor;
	}

	Tensor addReflectPad(Tensor input, py::tuple pad, const char *name)
	{
		int lpad = pad[0].cast<int>(), rpad = pad[1].cast<int>();

		auto plugin = m_pluginFactory.createReflectPad1D(lpad, rpad);
		auto reflectpad = m_graph->addPluginExt(&input.m_tensor, 1, *plugin);

		auto internalName = name + PluginFactory::reflectpad;
		reflectpad->setName(internalName.c_str());

		Tensor tensor = {reflectpad->getOutput(0)};
		return tensor;
	}
};


Graph *createNetwork(bool log)
{
	return new Graph(log);
}


enum class RTEngineType : int
{
	puzzle = 0,
	caffe = 1,
	onnx = 2
};


struct RTEngine
{
	ConsoleLogger m_logger;

	nv::IRuntime *m_infer = nullptr;
	nv::ICudaEngine *m_engine = nullptr;
	nv::IExecutionContext *m_context = nullptr;

	IPluginFactory *m_pluginFactory = nullptr;


	RTEngine(const std::string& path, RTEngineType enginetype, bool log) : m_logger(log)
	{
		std::ifstream file(path, std::ios::binary);
		if (!file.is_open())
			throw std::invalid_argument("Invalid engine path: " + path);

		std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		m_infer = nv::createInferRuntime(m_logger);
		if (m_infer == nullptr)
			throw std::runtime_error("Failed creating inference runtime");

		switch (enginetype)
		{
			case RTEngineType::puzzle:
				m_pluginFactory = new PluginFactory();
				break;

			case RTEngineType::caffe:
				m_pluginFactory = new CaffePluginFactory();
				break;

			case RTEngineType::onnx:
				break;
		}

		m_engine = m_infer->deserializeCudaEngine(&content[0], content.length(), m_pluginFactory);
		if (m_engine == nullptr)
			throw std::runtime_error("Failed creating engine");

		m_context = m_engine->createExecutionContext();
		if (m_context == nullptr)
			throw std::runtime_error("Failed creating context");
	}

	void enqueue(int batchSize, py::list bindings)
	{
		std::vector<void *> buffers;

		for (size_t i = 0; i < len(bindings); i++)
		{
			size_t binding = py::cast<size_t>(bindings[i]);
			buffers.push_back(reinterpret_cast<void *>(binding));
		}

		m_context->enqueue(batchSize, &buffers[0], nullptr, nullptr);
	}

	virtual ~RTEngine()
	{
		if (m_context != nullptr) m_context->destroy();
		if (m_engine != nullptr) m_engine->destroy();

		if (m_pluginFactory != nullptr) delete m_pluginFactory;
		if (m_infer != nullptr) m_infer->destroy();
	}
};


PYBIND11_MODULE(Driver, m)
{
	py::class_<nv::IInt8EntropyCalibrator2, ICalibrator,
		std::unique_ptr<nv::IInt8EntropyCalibrator2, py::nodelete>>(m, "ICalibrator")
		.def(py::init<>());

	m.def("buildRTEngineFromCaffe", &buildRTEngineFromCaffe);
	m.def("buildRTEngineFromOnnx", &buildRTEngineFromOnnx);

	py::enum_<DataType>(m, "DataType")
		.value("float", DataType::float_)
		.value("int8", DataType::int8)
		.value("half", DataType::half);

	py::enum_<ActivationType>(m, "ActivationType")
		.value("relu", ActivationType::relu)
		.value("leakyRelu", ActivationType::leakyRelu)
		.value("clip", ActivationType::clip)
		.value("sigmoid", ActivationType::sigmoid)
		.value("tanh", ActivationType::tanh);

	py::enum_<RNNMode>(m, "RNNMode")
		.value("relu", RNNMode::relu)
		.value("tanh", RNNMode::tanh)
		.value("lstm", RNNMode::lstm)
		.value("gru", RNNMode::gru);

	py::enum_<RNNDirection>(m, "RNNDirection")
		.value("uni", RNNDirection::uni)
		.value("bi", RNNDirection::bi);

	py::enum_<RNNInputMode>(m, "RNNInputMode")
		.value("linear", RNNInputMode::linear)
		.value("skip", RNNInputMode::skip);

	py::enum_<RNNGateType>(m, "RNNGateType")
		.value("input", RNNGateType::input)
		.value("output", RNNGateType::output)
		.value("forget", RNNGateType::forget)
		.value("update", RNNGateType::update)
		.value("reset", RNNGateType::reset)
		.value("cell", RNNGateType::cell)
		.value("hidden", RNNGateType::hidden);

	py::class_<Tensor>(m, "Tensor")
		.def_property_readonly("name", &Tensor::getName)
		.def_property_readonly("shape", &Tensor::getShape)
		.def("setName", &Tensor::setName);

	py::class_<Graph>(m, "Graph")
		.def("platformHasFastFp16", &Graph::platformHasFastFp16)
		.def("platformHasFastInt8", &Graph::platformHasFastInt8)
		.def("setFp16Mode", &Graph::setFp16Mode)
		.def("setInt8Mode", &Graph::setInt8Mode)
		.def("setInt8Calibrator", &Graph::setInt8Calibrator)

		.def("markOutput", &Graph::markOutput)
		.def("setMaxBatchSize", &Graph::setMaxBatchSize)
		.def("setMaxWorkspaceSize", &Graph::setMaxWorkspaceSize)
		.def("buildCudaEngine", &Graph::buildCudaEngine)

		.def("addInput", &Graph::addInput)
		.def("addConvolution", &Graph::addConvolution)
		.def("addScale", &Graph::addScale)
		.def("addActivation", &Graph::addActivation)
		.def("addPooling", &Graph::addPooling)
		.def("addCrossMapLRN", &Graph::addCrossMapLRN)
		.def("addAdd", &Graph::addAdd)
		.def("addConcatenation", &Graph::addConcatenation)
		.def("addFlatten", &Graph::addFlatten)
		.def("addLinear", &Graph::addLinear)
		.def("addSoftMax", &Graph::addSoftMax)
		.def("addSwapAxes", &Graph::addSwapAxes)
		.def("addMoveAxis", &Graph::addMoveAxis)
		.def("addSplit", &Graph::addSplit)
		.def("addReshape", &Graph::addReshape)
		.def("addGroupLinear", &Graph::addGroupLinear)
		.def("addSum", &Graph::addSum)
		.def("addRNN", &Graph::addRNN)
		.def("addUpsample", &Graph::addUpsample)
		.def("addPRelu", &Graph::addPRelu)
		.def("addReflectPad", &Graph::addReflectPad);

	m.def("createNetwork", &createNetwork, py::return_value_policy::take_ownership);

	py::enum_<RTEngineType>(m, "RTEngineType")
		.value("puzzle", RTEngineType::puzzle)
		.value("caffe", RTEngineType::caffe)
		.value("onnx", RTEngineType::onnx);

	py::class_<RTEngine>(m, "RTEngine")
		.def(py::init<const std::string&, RTEngineType, bool>())
		.def("enqueue", &RTEngine::enqueue);
}
