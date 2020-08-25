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


enum class RTDataType : int
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

	void setType(RTDataType dtype)
	{
		m_tensor->setType(static_cast<nv::DataType>(dtype));
	}

	RTDataType getType()
	{
		return static_cast<RTDataType>(m_tensor->getType());
	}
};


struct ConsoleLogger : nv::ILogger
{
	bool m_enabled;


	ConsoleLogger(bool enabled)
	{
		m_enabled = enabled;

		if (enabled)
		{
			std::cerr << "[TensorRT] LOGGER: Using version: ";
			std::cerr << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.';
			std::cerr << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << std::endl;
		}
	}


	void log(nv::ILogger::Severity severity, const char *msg) override
	{
		if (!m_enabled)
			return;

		std::cerr << "[TensorRT] ";

		switch (severity)
		{
			case Severity::kINTERNAL_ERROR:
			{
				std::cerr << "INTERNAL_ERROR: ";
				break;
			}
			case Severity::kERROR:
			{
				 std::cerr << "ERROR: ";
				 break;
			}
			case Severity::kWARNING:
			{
				std::cerr << "WARNING: ";
				break;
			}
			case Severity::kINFO:
			{
				std::cerr << "INFO: ";
				break;
			}
			case Severity::kVERBOSE:
			{
				std::cerr << "VERBOSE: ";
				break;
			}
			default:
			{
				std::cerr << "UNKNOWN: ";
				break;
			}
		}

		std::cerr << msg << std::endl;
	}
};


struct ICalibrator : nv::IInt8EntropyCalibrator2
{
	std::string m_cachename, m_cache;


	ICalibrator(const std::string &cachename) : m_cachename(cachename) {}

	int getBatchSize() const override
	{
		PYBIND11_OVERLOAD_PURE(int, nv::IInt8EntropyCalibrator2, getBatchSize);
	}

	bool getBatch(void *bindings[], const char *names[], int nbBindings) override
	{
		py::list pybindings, pynames;
		for (int i = 0; i < nbBindings; i += 1)
		{
			pybindings.append(reinterpret_cast<std::size_t>(&bindings[i]));
			pynames.append(names[i]);
		}

		PYBIND11_OVERLOAD_PURE(bool, nv::IInt8EntropyCalibrator2, getBatch, pybindings, pynames);
	}

	const void *readCalibrationCache(std::size_t &length) override
	{
		std::ifstream file(m_cachename, std::ios::binary);

		if (!file.is_open())
		{
			length = 0;
			return nullptr;
		}

		m_cache = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

		length = m_cache.size();
		return m_cache.c_str();
	}

	void writeCalibrationCache(const void *ptr, std::size_t length) override
	{
		std::ofstream file(m_cachename, std::ios::binary);

		if (file.is_open())
			file.write(reinterpret_cast<const char *>(ptr), length);
	}
};


template <typename T>
struct TRTInstance
{
	T instance;

	TRTInstance() : instance(nullptr) {}
	TRTInstance(T instance) : instance(instance) {}
	TRTInstance(TRTInstance<T> &other) = delete;
	TRTInstance(TRTInstance<T> &&other) : instance(other.instance) { other.instance = nullptr; }
	~TRTInstance() { if (instance != nullptr) instance->destroy(); }

	T get() { return instance; }
};


inline static void setBuilderConfigFlag(nv::IBuilderConfig &config, nv::BuilderFlag bit, bool mode)
{
	auto flags = config.getFlags();
	flags = mode ? (flags | 1U << static_cast<unsigned>(bit)) : (flags & ~(1U << static_cast<unsigned>(bit)));

	config.setFlags(flags);
}


void buildRTEngine(nvinfer1::INetworkDefinition *network, nv::IBuilder *builder, int batchsize, int workspace,
				   RTDataType mode, ICalibrator *calibrator, const std::string &savepath)
{
	builder->setMaxBatchSize(batchsize);

	auto configPtr = builder->createBuilderConfig();
	if (configPtr == nullptr)
		throw std::runtime_error("Failed to create builder config");

	TRTInstance<decltype(configPtr)> config(configPtr);
	config.get()->setMaxWorkspaceSize(workspace);

	if (mode == RTDataType::int8)
	{
		setBuilderConfigFlag(*config.get(), nv::BuilderFlag::kINT8, true);
		config.get()->setInt8Calibrator(calibrator);
	}
	else if (mode == RTDataType::half)
	{
		setBuilderConfigFlag(*config.get(), nv::BuilderFlag::kFP16, true);
	}

	auto enginePtr = builder->buildEngineWithConfig(*network, *config.get());
	if (enginePtr == nullptr)
		throw std::runtime_error("Failed to create engine");

	TRTInstance<decltype(enginePtr)> engine(enginePtr);

	auto streamPtr = engine.get()->serialize();
	if (streamPtr == nullptr)
		throw std::runtime_error("Failed to serialize engine");

	TRTInstance<decltype(streamPtr)> stream(streamPtr);

	std::ofstream file(savepath, std::ios::binary);
	if (!file.is_open())
		throw std::invalid_argument("Invalid engine save path: " + savepath);

	file.write(reinterpret_cast<char *>(stream.get()->data()), stream.get()->size());
}


void buildRTEngineFromCaffe(const std::string &prototxt, const std::string &caffemodel, int batchsize,
							py::list outlayers, RTDataType mode, ICalibrator *calibrator, int workspace,
							const std::string &savepath, bool log)
{
	ConsoleLogger logger(log);

	auto builderPtr = nv::createInferBuilder(logger);
	if (builderPtr == nullptr)
		throw std::runtime_error("Failed to create builder");

	TRTInstance<decltype(builderPtr)> builder(builderPtr);

	if (mode == RTDataType::half && !builder.get()->platformHasFastFp16())
		throw std::invalid_argument("Platform has no fast fp16 support");

	else if (mode == RTDataType::int8 && !builder.get()->platformHasFastInt8())
		throw std::invalid_argument("Platform has no fast int8 support");

	auto networkPtr = builder.get()->createNetworkV2(0);
	if (networkPtr == nullptr)
		throw std::runtime_error("Failed to create network");

	TRTInstance<decltype(networkPtr)> network(networkPtr);

	auto parserPtr = nvcaffeparser1::createCaffeParser();
	if (parserPtr == nullptr)
		throw std::runtime_error("Failed to create caffe parser");

	TRTInstance<decltype(parserPtr)> parser(parserPtr);

	auto blobNameToTensor = parser.get()->parse(
		prototxt.c_str(), caffemodel.c_str(), *network.get(), static_cast<nv::DataType>(mode)
	);

	if (blobNameToTensor == nullptr)
	{
		const char *msg = "failed to parse caffe file";
		logger.log(nvinfer1::ILogger::Severity::kERROR, msg);

		throw std::invalid_argument(msg);
	}

	for (std::size_t i = 0; i < len(outlayers); i += 1)
	{
		std::string outlayer = py::cast<std::string>(outlayers[i]);
		network.get()->markOutput(*blobNameToTensor->find(outlayer.c_str()));
	}

	buildRTEngine(network.get(), builder.get(), batchsize, workspace, mode, calibrator, savepath);
}


void buildRTEngineFromOnnx(const std::string &onnxname, int batchsize, RTDataType mode, ICalibrator *calibrator,
						   int workspace, const std::string &savepath, bool log)
{
	ConsoleLogger logger(log);

	auto builderPtr = nv::createInferBuilder(logger);
	if (builderPtr == nullptr)
		throw std::runtime_error("Failed to create builder");

	TRTInstance<decltype(builderPtr)> builder(builderPtr);

	if (mode == RTDataType::half && !builder.get()->platformHasFastFp16())
		throw std::invalid_argument("Platform has no fast fp16 support");

	else if (mode == RTDataType::int8 && !builder.get()->platformHasFastInt8())
		throw std::invalid_argument("Platform has no fast int8 support");

	auto networkPtr = builder.get()->createNetworkV2(0);
	if (networkPtr == nullptr)
		throw std::runtime_error("Failed to create network");

	TRTInstance<decltype(networkPtr)> network(networkPtr);

	auto parserPtr = nvonnxparser::createParser(*network.get(), logger);
	if (parserPtr == nullptr)
		throw std::runtime_error("Failed to create onnx parser");

	TRTInstance<decltype(parserPtr)> parser(parserPtr);

	if (!parser.get()->parseFromFile(onnxname.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
	{
		const char *msg = "failed to parse onnx file";
		logger.log(nvinfer1::ILogger::Severity::kERROR, msg);

		throw std::invalid_argument(msg);
	}

	buildRTEngine(network.get(), builder.get(), batchsize, workspace, mode, calibrator, savepath);
}


struct Graph
{
	ConsoleLogger m_logger;
	std::vector<TRTInstance<nv::IPluginV2 *>> m_plugins;

	TRTInstance<nv::IBuilder *> m_builder;
	TRTInstance<nv::IBuilderConfig *> m_config;
	TRTInstance<nv::INetworkDefinition *> m_graph;


	Graph(bool log) : m_logger(log)
	{
		auto builderPtr = nv::createInferBuilder(m_logger);
		if (builderPtr == nullptr)
			throw std::runtime_error("Failed to create builder");

		m_builder.instance = builderPtr;

		auto configPtr = m_builder.get()->createBuilderConfig();
		if (configPtr == nullptr)
			throw std::runtime_error("Failed to create builder config");

		m_config.instance = configPtr;

		auto graphPtr = m_builder.get()->createNetworkV2(0);
		if (graphPtr == nullptr)
			throw std::runtime_error("Failed to create graph");

		m_graph.instance = graphPtr;
	}

	bool platformHasFastFp16()
	{
		return m_builder.get()->platformHasFastFp16();
	}

	void setFp16Mode(bool mode)
	{
		setBuilderConfigFlag(*m_config.get(), nv::BuilderFlag::kFP16, mode);
	}

	bool platformHasFastInt8()
	{
		return m_builder.get()->platformHasFastInt8();
	}

	void setInt8Mode(bool mode)
	{
		setBuilderConfigFlag(*m_config.get(), nv::BuilderFlag::kINT8, mode);
	}

	void setInt8Calibrator(ICalibrator *calibrator)
	{
		m_config.get()->setInt8Calibrator(calibrator);
	}

	void markOutput(Tensor tensor)
	{
		m_graph.get()->markOutput(*tensor.m_tensor);
	}

	void setMaxBatchSize(int batchsize)
	{
		m_builder.get()->setMaxBatchSize(batchsize);
	}

	void setMaxWorkspaceSize(std::size_t size)
	{
		m_config.get()->setMaxWorkspaceSize(size);
	}

	void buildCudaEngine(const std::string &savepath)
	{
		auto enginePtr = m_builder.get()->buildEngineWithConfig(*m_graph.get(), *m_config.get());
		if (enginePtr == nullptr)
			throw std::runtime_error("Failed to create engine");

		TRTInstance<decltype(enginePtr)> engine(enginePtr);

		auto streamPtr = engine.get()->serialize();
		if (streamPtr == nullptr)
			throw std::runtime_error("Failed to serialize engine");

		TRTInstance<decltype(streamPtr)> stream(streamPtr);

		std::ofstream file(savepath, std::ios::binary);
		if (!file.is_open())
			throw std::invalid_argument("Invalid engine save path: " + savepath);

		file.write(reinterpret_cast<char *>(stream.get()->data()), stream.get()->size());
	}

	Tensor addInput(const char *name, RTDataType dtype, py::tuple shape)
	{
		nv::Dims dims;
		dims.nbDims = static_cast<int>(py::len(shape));

		for (int i = 0; i < dims.nbDims; i += 1)
			dims.d[i] = py::cast<int>(shape[i]);

		Tensor tensor = {m_graph.get()->addInput(name, static_cast<nv::DataType>(dtype), dims)};
		return tensor;
	}

	Tensor addConvolution(Tensor input, int outmaps, py::tuple kernel, std::size_t Wdata, int64_t Wlen,
						  std::size_t biasdata, int64_t biaslen, py::tuple stride, py::tuple pad, py::tuple pydilation,
						  py::tuple postpad, bool isDeconvolution, const char *name)
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
			auto deconv = m_graph.get()->addDeconvolutionNd(*input.m_tensor, outmaps, kernelSize, W, bias);
			deconv->setName(name);

			deconv->setStrideNd(striding);
			deconv->setPaddingNd(padding);

			auto postpadding = nv::DimsHW(py::cast<int>(postpad[0]), py::cast<int>(postpad[1]));

			if (postpadding.d[0] > 0 || postpadding.d[1] > 0)
			{
				auto prepadding = nv::DimsHW(0, 0);
				auto padlayer = m_graph.get()->addPaddingNd(*deconv->getOutput(0), prepadding, postpadding);

				layer = padlayer;
			}
			else
				layer = deconv;
		}
		else
		{
			auto conv = m_graph.get()->addConvolutionNd(*input.m_tensor, outmaps, kernelSize, W, bias);
			conv->setName(name);

			conv->setStrideNd(striding);
			conv->setPaddingNd(padding);

			auto dilation = nv::DimsHW(py::cast<int>(pydilation[0]), py::cast<int>(pydilation[1]));
			conv->setDilationNd(dilation);

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

		auto scale = m_graph.get()->addScale(*input.m_tensor, nv::ScaleMode::kCHANNEL, shift, scaling, power);
		scale->setName(name);

		Tensor tensor = {scale->getOutput(0)};
		return tensor;
	}

	Tensor addActivation(Tensor input, ActivationType type, float alpha, float beta, const char *name)
	{
		auto act = m_graph.get()->addActivation(*input.m_tensor, static_cast<nv::ActivationType>(type));

		act->setAlpha(alpha);
		act->setBeta(beta);
		act->setName(name);

		Tensor tensor = {act->getOutput(0)};
		return tensor;
	}

	Tensor addPooling(Tensor input, bool avg, py::tuple kernel, py::tuple stride, py::tuple pad, const char *name)
	{
		auto kernelSize = nv::DimsHW(py::cast<int>(kernel[0]), py::cast<int>(kernel[1]));

		auto pool = m_graph.get()->addPoolingNd(
			*input.m_tensor, avg ? nv::PoolingType::kAVERAGE : nv::PoolingType::kMAX, kernelSize
		);
		pool->setName(name);

		auto striding = nv::DimsHW(py::cast<int>(stride[0]), py::cast<int>(stride[1]));
		pool->setStrideNd(striding);

		auto padding = nv::DimsHW(py::cast<int>(pad[0]), py::cast<int>(pad[1]));
		pool->setPaddingNd(padding);

		Tensor tensor = {pool->getOutput(0)};
		return tensor;
	}

	Tensor addCrossMapLRN(Tensor input, int N, float alpha, float beta, float K, const char *name)
	{
		auto lrn = m_graph.get()->addLRN(*input.m_tensor, N, alpha, beta, K);
		lrn->setName(name);

		Tensor tensor = {lrn->getOutput(0)};
		return tensor;
	}

	Tensor addAdd(Tensor input1, Tensor input2, const char *name)
	{
		auto add = m_graph.get()->addElementWise(*input1.m_tensor, *input2.m_tensor, nv::ElementWiseOperation::kSUM);
		add->setName(name);

		Tensor tensor = {add->getOutput(0)};
		return tensor;
	}

	Tensor addConcatenation(py::list inputs, const char *name)
	{
		std::vector<nv::ITensor *> tensors(py::len(inputs));

		for (std::size_t i = 0; i < py::len(inputs); i += 1)
		{
			Tensor tensor = py::cast<Tensor>(inputs[i]);
			tensors[i] = tensor.m_tensor;
		}

		auto concat = m_graph.get()->addConcatenation(&tensors[0], static_cast<int>(tensors.size()));
		concat->setName(name);

		Tensor tensor = {concat->getOutput(0)};
		return tensor;
	}

	Tensor addFlatten(Tensor input, const char *name)
	{
		auto flatten = m_graph.get()->addShuffle(*input.m_tensor);
		flatten->setName(name);

		auto indims = input.m_tensor->getDimensions();

		nv::Dims outdims;
		outdims.nbDims = 1;
		outdims.d[0] = 1;

		for (int i = 0; i < indims.nbDims; i += 1)
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
		auto inReshape = m_graph.get()->addShuffle(*input.m_tensor);

		auto inReshapeDims = nv::Dims3(indims.d[0], 1, 1);
		inReshape->setReshapeDimensions(inReshapeDims);

		auto linear = m_graph.get()->addFullyConnected(*inReshape->getOutput(0), outputs, W, bias);
		linear->setName(name);

		auto outReshape = m_graph.get()->addShuffle(*linear->getOutput(0));

		nv::Dims outReshapeDims;
		outReshapeDims.nbDims = 1;
		outReshapeDims.d[0] = outputs;

		outReshape->setReshapeDimensions(outReshapeDims);

		Tensor tensor = {outReshape->getOutput(0)};
		return tensor;
	}

	Tensor addSoftMax(Tensor input, const char *name)
	{
		auto softmax = m_graph.get()->addSoftMax(*input.m_tensor);
		softmax->setName(name);

		Tensor tensor = {softmax->getOutput(0)};
		return tensor;
	}

	Tensor addSwapAxes(Tensor input, int axis1, int axis2, const char *name)
	{
		auto swapaxes = m_graph.get()->addShuffle(*input.m_tensor);
		swapaxes->setName(name);

		nv::Permutation permutation;
		for (int i = 0; i < nv::Dims::MAX_DIMS; i += 1)
			permutation.order[i] = i;

		permutation.order[axis1] = axis2;
		permutation.order[axis2] = axis1;

		swapaxes->setFirstTranspose(permutation);

		Tensor tensor = {swapaxes->getOutput(0)};
		return tensor;
	}

	Tensor addMoveAxis(Tensor input, int src, int dst, const char *name)
	{
		auto moveaxis = m_graph.get()->addShuffle(*input.m_tensor);
		moveaxis->setName(name);

		nv::Permutation permutation;
		for (int i = 0; i < nv::Dims::MAX_DIMS; i += 1)
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

		for (std::size_t i = 0; i < sections.size(); i += 1)
		{
			nv::Dims start, size, stride;

			start.nbDims = static_cast<int>(inshape.size());
			start.d[0] = offset;
			offset += sections[i];

			for (std::size_t d = 0; d < inshape.size() - 1; d += 1)
				start.d[d + 1] = 0;

			size.nbDims = static_cast<int>(inshape.size());
			size.d[0] = sections[i];
			for (std::size_t d = 0; d < inshape.size() - 1; d += 1)
				size.d[1 + d] = inshape[1 + d];

			stride.nbDims = static_cast<int>(inshape.size());
			for (std::size_t d = 0; d < inshape.size(); d += 1)
				stride.d[d] = 1;

			auto slice = m_graph.get()->addSlice(*input.m_tensor, start, size, stride);

			auto sliceName = std::string(name) + "_slice_" + std::to_string(i);
			slice->setName(sliceName.c_str());

			Tensor tensor = {slice->getOutput(0)};
			tensors[i] = tensor;
		}

		return tensors;
	}

	Tensor addReshape(Tensor input, py::tuple shape, const char *name)
	{
		auto reshape = m_graph.get()->addShuffle(*input.m_tensor);
		reshape->setName(name);

		nv::Dims dims;
		dims.nbDims = static_cast<int>(py::len(shape));

		for (int i = 0; i < dims.nbDims; i += 1)
			dims.d[i] = py::cast<int>(shape[i]);

		reshape->setReshapeDimensions(dims);

		Tensor tensor = {reshape->getOutput(0)};
		return tensor;
	}

	Tensor addGroupLinear(Tensor input, int groups, int insize, int outsize, std::size_t Wdata, int64_t Wlen,
						  std::size_t biasdata, int64_t biaslen, const char *name)
	{
		auto dims = nv::Dims2(insize, outsize);

		nv::Weights W = {nv::DataType::kFLOAT, reinterpret_cast<void *>(Wdata), Wlen};
		auto weights = m_graph.get()->addConstant(dims, W);

		nv::ILayer *linear = m_graph.get()->addMatrixMultiply(
			*input.m_tensor, nv::MatrixOperation::kNONE, *weights->getOutput(0), nv::MatrixOperation::kNONE
		);
		linear->setName(name);

		if (biasdata != 0)
		{
			auto biasDims = nv::Dims2(groups, outsize);

			nv::Weights b = {nv::DataType::kFLOAT, reinterpret_cast<void *>(biasdata), biaslen};
			auto bias = m_graph.get()->addConstant(biasDims, b);

			linear = m_graph.get()->addElementWise(
				*linear->getOutput(0), *bias->getOutput(0), nv::ElementWiseOperation::kSUM
			);
		}

		Tensor tensor = {linear->getOutput(0)};
		return tensor;
	}

	Tensor addSum(Tensor input, int axis, const char *name)
	{
		auto sum = m_graph.get()->addReduce(*input.m_tensor, nv::ReduceOperation::kSUM, axis, false);
		sum->setName(name);

		Tensor tensor = {sum->getOutput(0)};
		return tensor;
	}

	void updateRNNParams(nv::IRNNv2Layer *rnn, int layers,
						 const std::vector<std::size_t> &Wdata, const std::vector<int64_t> &Wlen,
						 const std::vector<std::size_t> &biasdata, const std::vector<int64_t> &biaslen)
	{
		const int keys = 2;

		for (int layer = 0; layer < layers; layer += 1)
		{
			for (int k = 0; k < keys; k += 1)
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
						  const std::vector<std::size_t> &Wdata, const std::vector<int64_t> &Wlen,
						  const std::vector<std::size_t> &biasdata, const std::vector<int64_t> &biaslen)
	{
		const int keys = 8;

		for (int layer = 0; layer < layers; layer += 1)
		{
			for (int k = 0; k < keys; k += 1)
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
						 const std::vector<std::size_t> &Wdata, const std::vector<int64_t> &Wlen,
						 const std::vector<std::size_t> &biasdata, const std::vector<int64_t> &biaslen)
	{
		const int keys = 6;

		for (int layer = 0; layer < layers; layer += 1)
		{
			for (int k = 0; k < keys; k += 1)
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
		auto rnn = m_graph.get()->addRNNv2(*input.m_tensor, layers, hsize, seqlen, static_cast<nv::RNNOperation>(mode));
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
		auto upsample = m_graph.get()->addResize(*input.m_tensor);
		upsample->setName(name);

		auto dims = input.m_tensor->getDimensions();

		std::vector<float> scales(dims.nbDims, static_cast<float>(scale));
		scales[0] = 1.0f;

		upsample->setScales(&scales[0], dims.nbDims);

		Tensor tensor = {upsample->getOutput(0)};
		return tensor;
	}

	Tensor addPRelu(Tensor input, std::size_t slopedata, int slopelen, const char *name)
	{
		auto dims = input.m_tensor->getDimensions();

		for (int i = 1; i < dims.nbDims; i += 1)
			dims.d[i] = 1;

		nv::Weights s = {nv::DataType::kFLOAT, reinterpret_cast<void *>(slopedata), slopelen};
		auto slope = m_graph.get()->addConstant(dims, s);

		auto prelu = m_graph.get()->addParametricReLU(*input.m_tensor, *slope->getOutput(0));
		prelu->setName(name);

		Tensor tensor = {prelu->getOutput(0)};
		return tensor;
	}

	Tensor addReflectPad(Tensor input, py::tuple pad, const char *name)
	{
		auto creator = getPluginRegistry()->getPluginCreator(
			PuzzlePluginCreator::reflectPad1DName, PuzzlePluginCreator::version
		);

		auto padding = nv::Dims2(py::cast<int>(pad[0]), py::cast<int>(pad[1]));

		nv::PluginField padfield = {"pad", reinterpret_cast<const void *>(&padding), nv::PluginFieldType::kDIMS, 1};
		nv::PluginFieldCollection fc = {1, &padfield};

		auto pluginPtr = creator->createPlugin(name, &fc);
		TRTInstance<decltype(pluginPtr)> plugin(pluginPtr);

		auto reflectpad = m_graph.get()->addPluginV2(&input.m_tensor, 1, *plugin.get());
		reflectpad->setName(name);

		m_plugins.push_back(std::move(plugin));
		Tensor tensor = {reflectpad->getOutput(0)};

		return tensor;
	}

	Tensor addInstanceNorm(Tensor input, std::size_t scaledata, std::size_t biasdata, int32_t length, float epsilon,
						   const char *name)
	{
		auto creator = getPluginRegistry()->getPluginCreator(
			PuzzlePluginCreator::instNorm2DName, PuzzlePluginCreator::version
		);

		nv::PluginField fields[] = {
			{"scale", reinterpret_cast<const void *>(scaledata), nv::PluginFieldType::kFLOAT32, length},
			{"bias", reinterpret_cast<const void *>(biasdata), nv::PluginFieldType::kFLOAT32, length},
			{"epsilon", reinterpret_cast<const void *>(&epsilon), nv::PluginFieldType::kFLOAT32, 1}
		};
		nv::PluginFieldCollection fc = {3, fields};

		auto pluginPtr = creator->createPlugin(name, &fc);
		TRTInstance<decltype(pluginPtr)> plugin(pluginPtr);

		auto instnorm = m_graph.get()->addPluginV2(&input.m_tensor, 1, *plugin.get());
		instnorm->setName(name);

		m_plugins.push_back(std::move(plugin));
		Tensor tensor = {instnorm->getOutput(0)};

		return tensor;
	}
};


Graph *createNetwork(bool log)
{
	return new Graph(log);
}


struct RTEngine
{
	ConsoleLogger m_logger;

	TRTInstance<nv::IRuntime *> m_infer;
	TRTInstance<nv::ICudaEngine *> m_engine;
	TRTInstance<nv::IExecutionContext *> m_context;


	RTEngine(const std::string &path, bool log) : m_logger(log)
	{
		auto inferPtr = nv::createInferRuntime(m_logger);
		if (inferPtr == nullptr)
			throw std::runtime_error("Failed creating inference runtime");

		m_infer.instance = inferPtr;

		std::ifstream file(path, std::ios::binary);
		if (!file.is_open())
			throw std::invalid_argument("Invalid engine path: " + path);

		std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		auto enginePtr = m_infer.get()->deserializeCudaEngine(&content[0], content.length(), nullptr);
		if (enginePtr == nullptr)
			throw std::runtime_error("Failed creating engine");

		m_engine.instance = enginePtr;

		auto contextPtr = m_engine.get()->createExecutionContext();
		if (contextPtr == nullptr)
			throw std::runtime_error("Failed creating context");

		m_context.instance = contextPtr;
	}

	int getBatchSize()
	{
		return m_engine.get()->getMaxBatchSize();
	}

	std::vector<std::vector<int>> getInshape()
	{
		int nbindings = m_engine.get()->getNbBindings();
		std::vector<std::vector<int>> inshape;

		for (int i = 0; i < nbindings; i += 1)
		{
			if (!m_engine.get()->bindingIsInput(i))
				continue;

			auto dims = m_engine.get()->getBindingDimensions(i);
			inshape.push_back(std::vector<int>(dims.d, dims.d + dims.nbDims));
		}

		return inshape;
	}

	std::vector<std::vector<int>> getOutshape()
	{
		int nbindings = m_engine.get()->getNbBindings();
		std::vector<std::vector<int>> outshape;

		for (int i = 0; i < nbindings; i += 1)
		{
			if (m_engine.get()->bindingIsInput(i))
				continue;

			auto dims = m_engine.get()->getBindingDimensions(i);
			outshape.push_back(std::vector<int>(dims.d, dims.d + dims.nbDims));
		}

		return outshape;
	}

	void enqueue(int batchSize, py::list bindings)
	{
		std::vector<void *> buffers;

		for (size_t i = 0; i < len(bindings); i += 1)
		{
			size_t binding = py::cast<size_t>(bindings[i]);
			buffers.push_back(reinterpret_cast<void *>(binding));
		}

		m_context.get()->enqueue(batchSize, &buffers[0], nullptr, nullptr);
	}
};


PYBIND11_MODULE(Driver, m)
{
	py::class_<nv::IInt8EntropyCalibrator2, ICalibrator,
		std::unique_ptr<nv::IInt8EntropyCalibrator2, py::nodelete>>(m, "ICalibrator")
		.def(py::init<const std::string &>());

	m.def("buildRTEngineFromCaffe", &buildRTEngineFromCaffe);
	m.def("buildRTEngineFromOnnx", &buildRTEngineFromOnnx);

	py::enum_<RTDataType>(m, "RTDataType")
		.value("float", RTDataType::float_)
		.value("int8", RTDataType::int8)
		.value("half", RTDataType::half);

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
		.def_property_readonly("dtype", &Tensor::getType)
		.def("setName", &Tensor::setName)
		.def("setType", &Tensor::setType);

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
		.def("addReflectPad", &Graph::addReflectPad)
		.def("addInstanceNorm", &Graph::addInstanceNorm);

	m.def("createNetwork", &createNetwork, py::return_value_policy::take_ownership);

	py::class_<RTEngine>(m, "RTEngine")
		.def(py::init<const std::string &, bool>())
		.def_property_readonly("batchsize", &RTEngine::getBatchSize)
		.def_property_readonly("inshape", &RTEngine::getInshape)
		.def_property_readonly("outshape", &RTEngine::getOutshape)
		.def("enqueue", &RTEngine::enqueue);
}
