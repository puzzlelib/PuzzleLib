#ifdef _WIN32
	#pragma warning(push)
	#pragma warning(disable : 4244 4251 4275)
#endif

#include <ie_builders.hpp>
#include <inference_engine.hpp>
namespace ie = InferenceEngine;

#ifdef _WIN32
	#pragma warning(pop)
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;


enum class ActivationType : int
{
	relu,
	sigmoid
};


struct Tensor
{
	ie::Builder::Network *m_graph;

	ie::idx_t m_index;
	std::size_t m_offset;


	std::string getName()
	{
		return m_graph->getLayer(m_index)->getName();
	}

	ie::Port getPort()
	{
		return m_graph->getLayer(m_index)->getOutputPorts()[m_offset];
	}

	ie::SizeVector getShape()
	{
		return getPort().shape();
	}
};


struct Graph
{
	ie::Builder::Network m_graph;


	Graph(const std::string &name) : m_graph(name) {}

	void build(const char *xmlpath, const char *binpath)
	{
		auto net = m_graph.build();
		auto cnn = ie::Builder::convertToICNNNetwork(net);

		cnn->serialize(xmlpath, binpath, nullptr);
	}

	void markOutput(Tensor input, const char *name)
	{
		assert(input.m_offset == 0);

		m_graph.getLayer(input.m_index)->setName(name);
		m_graph.addLayer({input.m_index}, ie::Builder::OutputLayer("outdata").setPort(input.getPort()));
	}

	Tensor addInput(const char *name, py::tuple shape)
	{
		ie::SizeVector dims(py::len(shape) + 1);
		dims[0] = 1;

		for (std::size_t i = 0; i < py::len(shape); i += 1)
			dims[i + 1] = py::cast<std::size_t>(shape[i]);

		auto index = m_graph.addLayer(ie::Builder::InputLayer(name).setPort(ie::Port(dims)));

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addConvolution(Tensor input, ie::SizeVector shape, py::tuple kernel,
						  std::size_t Wdata, std::size_t biasdata, py::tuple stride, py::tuple pad, const char *name)
	{
		auto conv = ie::Builder::ConvolutionLayer(name);
		conv.setOutDepth(shape[1]);

		auto inshape = input.getShape();
		conv.setInputPort(ie::Port(inshape)).setOutputPort(ie::Port(shape));

		std::vector<std::size_t> Wshape(py::len(kernel));
		for (std::size_t i = 0; i < Wshape.size(); i += 1)
			Wshape[i] = py::cast<std::size_t>(kernel[i]);

		conv.setKernel(Wshape);

		std::vector<std::size_t> strides(py::len(stride));
		for (std::size_t i = 0; i < stride.size(); i += 1)
			strides[i] = py::cast<std::size_t>(stride[i]);

		conv.setStrides(strides);

		std::vector<std::size_t> paddings(py::len(pad));
		for (std::size_t i = 0; i < pad.size(); i += 1)
			paddings[i] = py::cast<std::size_t>(pad[i]);

		conv.setPaddingsBegin(paddings);
		conv.setPaddingsEnd(paddings);

		auto Wblob = ie::make_shared_blob<float>(
			ie::TensorDesc(ie::Precision::FP32, {shape[1], inshape[1], Wshape[0], Wshape[1]}, ie::Layout::NCHW)
		);
		Wblob->allocate();

		auto WblobPtr = Wblob->buffer().as<void *>(), Wptr = reinterpret_cast<void *>(Wdata);
		std::memcpy(WblobPtr, Wptr, Wblob->byteSize());

		auto Windex = m_graph.addLayer(ie::Builder::ConstLayer("W").setData(Wblob));
		auto index = m_graph.addLayer(conv);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({Windex}, {index, 1});

		if (biasdata != 0)
		{
			auto biasBlob = ie::make_shared_blob<float>(ie::TensorDesc(ie::Precision::FP32, {shape[1]}, ie::Layout::C));
			biasBlob->allocate();

			auto biasBlobPtr = biasBlob->buffer().as<void *>(), biasptr = reinterpret_cast<void *>(biasdata);
			std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

			auto biasIndex = m_graph.addLayer(ie::Builder::ConstLayer("bias").setData(biasBlob));
			m_graph.connect({biasIndex}, {index, 2});
		}

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addScale(Tensor input, std::size_t maps, std::size_t scaledata, std::size_t biasdata, const char *name)
	{
		auto scale = ie::Builder::ScaleShiftLayer(name).setPort(input.getPort());

		auto scaleBlob = ie::make_shared_blob<float>(ie::TensorDesc(ie::Precision::FP32, {maps}, ie::Layout::C));
		scaleBlob->allocate();

		auto scaleBlobPtr = scaleBlob->buffer().as<void *>(), scaleptr = reinterpret_cast<void *>(scaledata);
		std::memcpy(scaleBlobPtr, scaleptr, scaleBlob->byteSize());

		auto biasBlob = ie::make_shared_blob<float>(ie::TensorDesc{ie::Precision::FP32, {maps}, ie::Layout::C});
		biasBlob->allocate();

		auto biasBlobPtr = biasBlob->buffer().as<void *>(), biasptr = reinterpret_cast<void *>(biasdata);
		std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

		auto scaleIndex = m_graph.addLayer(ie::Builder::ConstLayer("scale").setData(scaleBlob));
		auto biasIndex = m_graph.addLayer(ie::Builder::ConstLayer("bias").setData(biasBlob));
		auto index = m_graph.addLayer(scale);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({scaleIndex}, {index, 1});
		m_graph.connect({biasIndex}, {index, 2});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addActivation(Tensor input, ActivationType type, float alpha, const char *name)
	{
		auto port = input.getPort();
		ie::idx_t index = 0;

		switch (type)
		{
			case ActivationType::relu:
			{
				auto relu = ie::Builder::ReLULayer(name).setPort(port);

				if (alpha != 0.0f)
					relu.setNegativeSlope(alpha);

				index = m_graph.addLayer(relu);
				break;
			}

			case ActivationType::sigmoid:
			{
				index = m_graph.addLayer(ie::Builder::SigmoidLayer(name).setPort(port));
				break;
			}

			default:
				assert(false);
		}

		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addPooling(Tensor input, ie::SizeVector shape, bool avg, py::tuple kernel, py::tuple stride, py::tuple pad,
					  const char *name)
	{
		auto pool = ie::Builder::PoolingLayer(name).setInputPort(input.getPort()).setOutputPort(ie::Port(shape));

		pool.setPoolingType(
			avg ? ie::Builder::PoolingLayer::PoolingType::AVG : ie::Builder::PoolingLayer::PoolingType::MAX
		);
		pool.setRoundingType(ie::Builder::PoolingLayer::RoundingType::FLOOR);

		std::vector<std::size_t> Wshape(py::len(kernel));
		for (std::size_t i = 0; i < Wshape.size(); i += 1)
			Wshape[i] = py::cast<std::size_t>(kernel[i]);

		pool.setKernel(Wshape);

		std::vector<std::size_t> strides(py::len(stride));
		for (std::size_t i = 0; i < stride.size(); i += 1)
			strides[i] = py::cast<std::size_t>(stride[i]);

		pool.setStrides(strides);

		std::vector<std::size_t> paddings(py::len(pad));
		for (std::size_t i = 0; i < pad.size(); i += 1)
			paddings[i] = py::cast<std::size_t>(pad[i]);

		pool.setPaddingsBegin(paddings);
		pool.setPaddingsEnd(paddings);

		auto index = m_graph.addLayer(pool);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addAdd(Tensor input1, Tensor input2, const char *name)
	{
		auto add = ie::Builder::EltwiseLayer(name).setInputPorts({input1.getPort(), input2.getPort()});

		add.setOutputPort(input1.getPort());
		add.setEltwiseType(ie::Builder::EltwiseLayer::EltwiseType::SUM);

		auto index = m_graph.addLayer(add);
		m_graph.connect({input1.m_index, input1.m_offset}, {index, 0});
		m_graph.connect({input2.m_index, input2.m_offset}, {index, 1});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addConcat(py::list inputs, ie::SizeVector shape, const char *name)
	{
		std::vector<Tensor> tensors(py::len(inputs));
		std::vector<ie::Port> ports(py::len(inputs));

		for (std::size_t i = 0; i < py::len(inputs); i += 1)
		{
			Tensor tensor = py::cast<Tensor>(inputs[i]);

			tensors[i] = tensor;
			ports[i] = tensor.getPort();
		}

		auto concat = ie::Builder::ConcatLayer(name).setAxis(1).setOutputPort(ie::Port(shape));
		concat.setInputPorts(ports);

		auto index = m_graph.addLayer(concat);

		for (std::size_t i = 0; i < py::len(inputs); i += 1)
			m_graph.connect({tensors[i].m_index, tensors[i].m_offset}, {index, i});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addFlatten(Tensor input, ie::SizeVector shape, const char *name)
	{
		auto reshape = ie::Builder::ReshapeLayer(name).setDims({0, -1});
		reshape.setInputPort(input.getPort()).setOutputPort(ie::Port(shape));

		auto index = m_graph.addLayer(reshape);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addLinear(Tensor input, ie::SizeVector shape, std::size_t Wdata, std::size_t biasdata, const char *name)
	{
		auto inshape = input.getShape();

		auto Wblob = ie::make_shared_blob<float>(ie::TensorDesc(
			ie::Precision::FP32, {shape[1], inshape[1]}, ie::Layout::NC
		));
		Wblob->allocate();

		auto WblobPtr = Wblob->buffer().as<void *>(), Wptr = reinterpret_cast<void *>(Wdata);
		std::memcpy(WblobPtr, Wptr, Wblob->byteSize());

		auto linear = ie::Builder::FullyConnectedLayer(name).setOutputNum(shape[1]);
		linear.setInputPort(ie::Port(inshape)).setOutputPort(ie::Port(shape));

		auto Windex = m_graph.addLayer(ie::Builder::ConstLayer("W").setData(Wblob));
		auto index = m_graph.addLayer(linear);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({Windex}, {index, 1});

		if (biasdata != 0)
		{
			auto biasBlob = ie::make_shared_blob<float>(ie::TensorDesc(ie::Precision::FP32, {shape[1]}, ie::Layout::C));
			biasBlob->allocate();

			auto biasBlobPtr = biasBlob->buffer().as<void *>(), biasptr = reinterpret_cast<void *>(biasdata);
			std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

			auto biasIndex = m_graph.addLayer(ie::Builder::ConstLayer("bias").setData(biasBlob));
			m_graph.connect({biasIndex}, {index, 2});
		}

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	Tensor addSoftMax(Tensor input, const char *name)
	{
		auto index = m_graph.addLayer(ie::Builder::SoftMaxLayer(name).setAxis(1).setPort(input.getPort()));
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}

	std::vector<Tensor> addSplit(Tensor input, std::size_t axis, std::vector<ie::SizeVector> shapes, const char *name)
	{
		assert(axis == 1);

		auto split = ie::Builder::SplitLayer(name).setAxis(axis);
		split.setInputPort(input.getPort());

		std::vector<ie::Port> ports(shapes.size());

		for (std::size_t i = 0; i < shapes.size(); i += 1)
			ports[i] = ie::Port(shapes[i]);

		split.setOutputPorts(ports);

		auto index = m_graph.addLayer(split);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		std::vector<Tensor> tensors(shapes.size());

		for (std::size_t i = 0; i < shapes.size(); i += 1)
			tensors[i] = Tensor{&m_graph, index, i};

		return tensors;
	}

	Tensor addUpsample(Tensor input, int scale, const char *name)
	{
		auto resample = ie::Builder::ResampleLayer(name).setResampleType("caffe.ResampleParameter.NEAREST");
		resample.setFactor(static_cast<float>(scale));

		auto shape = input.getShape();
		resample.setInputPort(ie::Port(shape));

		for (std::size_t i = 2; i < shape.size(); i += 1)
			shape[i] *= scale;

		resample.setOutputPort(ie::Port(shape));

		auto index = m_graph.addLayer(resample);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index, 0};
		return tensor;
	}
};


Graph *createNetwork(const std::string &name)
{
	return new Graph(name);
}


struct VINOEngine
{
	ie::ExecutableNetwork m_engine;


	VINOEngine(std::size_t batchsize, const std::string &xmlpath, const std::string &binpath,
			   const std::string &backend)
	{
		ie::CNNNetReader reader;

		reader.ReadNetwork(xmlpath);
		reader.ReadWeights(binpath);

		ie::CNNNetwork net = reader.getNetwork();
		net.setBatchSize(batchsize);

		ie::Core core;

		if (backend == "CPU")
		{
#ifdef _WIN32
			const char *extlib = "cpu_extension_avx2.dll";
#else
			const char *extlib = "libcpu_extension_avx2.so";
#endif

			auto extension = ie::make_so_pointer<ie::IExtension>(extlib);
			core.AddExtension(extension, backend);
		}

		m_engine = core.LoadNetwork(net, backend);
	}

	py::dict getInshape()
	{
		py::dict inshape;

		for (auto &info : m_engine.GetInputsInfo())
			inshape[info.first.c_str()] = info.second->getTensorDesc().getDims();

		return inshape;
	}

	py::dict getOutshape()
	{
		py::dict outshape;

		for (auto &info : m_engine.GetOutputsInfo())
			outshape[info.first.c_str()] = info.second->getTensorDesc().getDims();

		return outshape;
	}

	void infer(py::dict outputs, py::dict inputs)
	{
		auto request = m_engine.CreateInferRequest();

		for (auto &info : m_engine.GetInputsInfo())
		{
			py::tuple input = inputs[info.first.c_str()];

			auto ptr = reinterpret_cast<float *>(py::cast<std::size_t>(input[0]));
			auto size = py::cast<std::size_t>(input[1]);

			auto blob = ie::make_shared_blob<float>(info.second->getTensorDesc(), ptr, size);
			request.SetBlob(info.first, blob);
		}

		request.Infer();

		for (auto &info : m_engine.GetOutputsInfo())
		{
			auto blob = request.GetBlob(info.first);
			auto ptr = blob->buffer().as<void *>();

			py::tuple output = outputs[info.first.c_str()];

			auto outptr = reinterpret_cast<void *>(py::cast<std::size_t>(output[0]));
			auto size = py::cast<std::size_t>(output[1]);

			std::memcpy(outptr, ptr, size);
		}
	}
};


PYBIND11_MODULE(Driver, m)
{
	py::enum_<ActivationType>(m, "ActivationType")
		.value("relu", ActivationType::relu)
		.value("sigmoid", ActivationType::sigmoid);

	py::class_<Tensor>(m, "Tensor")
		.def_property_readonly("name", &Tensor::getName)
		.def_property_readonly("shape", &Tensor::getShape);

	py::class_<Graph>(m, "Graph")
		.def("markOutput", &Graph::markOutput)
		.def("build", &Graph::build)

		.def("addInput", &Graph::addInput)
		.def("addConvolution", &Graph::addConvolution)
		.def("addScale", &Graph::addScale)
		.def("addActivation", &Graph::addActivation)
		.def("addPooling", &Graph::addPooling)
		.def("addAdd", &Graph::addAdd)
		.def("addConcat", &Graph::addConcat)
		.def("addFlatten", &Graph::addFlatten)
		.def("addLinear", &Graph::addLinear)
		.def("addSoftmax", &Graph::addSoftMax)
		.def("addSplit", &Graph::addSplit)
		.def("addUpsample", &Graph::addUpsample);

	m.def("createNetwork", &createNetwork, py::return_value_policy::take_ownership);

	py::class_<VINOEngine>(m, "VINOEngine")
		.def(py::init<std::size_t, const std::string &, const std::string &, const std::string &>())
		.def_property_readonly("inshape", &VINOEngine::getInshape)
		.def_property_readonly("outshape", &VINOEngine::getOutshape)
		.def("infer", &VINOEngine::infer);
}
