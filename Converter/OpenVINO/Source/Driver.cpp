#ifdef _WIN32
	#pragma warning(push)
	#pragma warning(disable : 4251)
	#pragma warning(disable : 4275)
	#pragma warning(disable : 4244)
#endif

#include <ie_builders.hpp>
#include <inference_engine.hpp>
using namespace InferenceEngine;

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
	Builder::Network *m_graph;

	idx_t m_index;
	std::size_t m_offset;


	std::string getName()
	{
		return m_graph->getLayer(m_index)->getName();
	}

	SizeVector getShape()
	{
		return m_graph->getLayer(m_index)->getOutputPorts()[m_offset].shape();
	}
};


struct Graph
{
	Builder::Network m_graph;


	Graph(std::string name) : m_graph(name) {}

	void build(const char *xmlpath, const char *binpath)
	{
		auto net = m_graph.build();
		auto cnn = InferenceEngine::Builder::convertToICNNNetwork(net);

		cnn->serialize(xmlpath, binpath, nullptr);
	}

	void markOutput(Tensor input, const char *name)
	{
		assert(input.m_offset == 0);

		m_graph.getLayer(input.m_index)->setName(name);
		m_graph.addLayer({input.m_index}, Builder::OutputLayer("outdata").setPort(Port(input.getShape())));
	}

	Tensor addInput(const char *name, py::tuple shape)
	{
		SizeVector dims(py::len(shape) + 1);
		dims[0] = 1;

		for (size_t i = 0; i < py::len(shape); i++)
			dims[i + 1] = py::cast<size_t>(shape[i]);

		idx_t index = m_graph.addLayer(Builder::InputLayer(name).setPort(Port(dims)));

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addConvolution(Tensor input, SizeVector shape, py::tuple kernel,
						  std::size_t Wdata, std::size_t biasdata, py::tuple stride, py::tuple pad, const char *name)
	{
		auto conv = Builder::ConvolutionLayer(name);
		conv.setOutDepth(shape[1]);

		SizeVector inshape = input.getShape();
		conv.setInputPort(Port(inshape)).setOutputPort(Port(shape));

		std::vector<std::size_t> Wshape(py::len(kernel));
		for (size_t i = 0; i < Wshape.size(); i++)
			Wshape[i] = py::cast<size_t>(kernel[i]);

		conv.setKernel(Wshape);

		std::vector<std::size_t> strides(py::len(stride));
		for (size_t i = 0; i < stride.size(); i++)
			strides[i] = py::cast<size_t>(stride[i]);

		conv.setStrides(strides);

		std::vector<std::size_t> paddings(py::len(pad));
		for (size_t i = 0; i < pad.size(); i++)
			paddings[i] = py::cast<size_t>(pad[i]);

		conv.setPaddingsBegin(paddings);
		conv.setPaddingsEnd(paddings);

		auto Wblob = make_shared_blob<float>(
			TensorDesc(Precision::FP32, {shape[1], inshape[1], Wshape[0], Wshape[1]}, Layout::NCHW)
		);
		Wblob->allocate();

		void *WblobPtr = Wblob->buffer().as<void *>(), *Wptr = reinterpret_cast<void *>(Wdata);
		std::memcpy(WblobPtr, Wptr, Wblob->byteSize());

		idx_t Windex = m_graph.addLayer(Builder::ConstLayer("W").setData(Wblob));
		idx_t index = m_graph.addLayer(conv);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({Windex}, {index, 1});

		if (biasdata != 0)
		{
			auto biasBlob = make_shared_blob<float>(TensorDesc(Precision::FP32, {shape[1]}, Layout::C));
			biasBlob->allocate();

			void *biasBlobPtr = biasBlob->buffer().as<void *>(), *biasptr = reinterpret_cast<void *>(biasdata);
			std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

			idx_t biasIndex = m_graph.addLayer(Builder::ConstLayer("bias").setData(biasBlob));
			m_graph.connect({biasIndex}, {index, 2});
		}

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addScale(Tensor input, std::size_t maps, std::size_t scaledata, std::size_t biasdata, const char *name)
	{
		auto scale = Builder::ScaleShiftLayer(name).setPort(Port(input.getShape()));

		auto scaleBlob = make_shared_blob<float>(TensorDesc(Precision::FP32, {maps}, Layout::C));
		scaleBlob->allocate();

		void *scaleBlobPtr = scaleBlob->buffer().as<void *>(), *scaleptr = reinterpret_cast<void *>(scaledata);
		std::memcpy(scaleBlobPtr, scaleptr, scaleBlob->byteSize());

		auto biasBlob = make_shared_blob<float>(TensorDesc{Precision::FP32, {maps}, Layout::C});
		biasBlob->allocate();

		void *biasBlobPtr = biasBlob->buffer().as<void *>(), *biasptr = reinterpret_cast<void *>(biasdata);
		std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

		idx_t scaleIndex = m_graph.addLayer(Builder::ConstLayer("scale").setData(scaleBlob));
		idx_t biasIndex = m_graph.addLayer(Builder::ConstLayer("bias").setData(biasBlob));
		idx_t index = m_graph.addLayer(scale);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({scaleIndex}, {index, 1});
		m_graph.connect({biasIndex}, {index, 2});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addActivation(Tensor input, ActivationType type, float alpha, const char *name)
	{
		SizeVector shape = input.getShape();
		idx_t index = 0;

		switch (type)
		{
			case ActivationType::relu:
			{
				auto relu = Builder::ReLULayer(name).setPort(Port(shape));

				if (alpha != 0.0f)
					relu.setNegativeSlope(alpha);

				index = m_graph.addLayer(relu);
				break;
			}

			case ActivationType::sigmoid:
			{
				index = m_graph.addLayer(Builder::SigmoidLayer(name).setPort(Port(shape)));
				break;
			}

			default:
				assert(false);
		}

		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addPooling(Tensor input, SizeVector shape, bool avg, py::tuple kernel, py::tuple stride, py::tuple pad,
					  const char *name)
	{
		auto pool = Builder::PoolingLayer(name).setInputPort(Port(input.getShape())).setOutputPort(Port(shape));

		pool.setPoolingType(avg ? Builder::PoolingLayer::PoolingType::AVG : Builder::PoolingLayer::PoolingType::MAX);
		pool.setRoundingType(Builder::PoolingLayer::RoundingType::FLOOR);

		std::vector<std::size_t> Wshape(py::len(kernel));
		for (size_t i = 0; i < Wshape.size(); i++)
			Wshape[i] = py::cast<size_t>(kernel[i]);

		pool.setKernel(Wshape);

		std::vector<std::size_t> strides(py::len(stride));
		for (size_t i = 0; i < stride.size(); i++)
			strides[i] = py::cast<size_t>(stride[i]);

		pool.setStrides(strides);

		std::vector<std::size_t> paddings(py::len(pad));
		for (size_t i = 0; i < pad.size(); i++)
			paddings[i] = py::cast<size_t>(pad[i]);

		pool.setPaddingsBegin(paddings);
		pool.setPaddingsEnd(paddings);

		idx_t index = m_graph.addLayer(pool);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addAdd(Tensor input1, Tensor input2, const char *name)
	{
		Port port = Port(input1.getShape());
		auto add = Builder::EltwiseLayer(name).setInputPorts({port, Port(input2.getShape())});

		add.setOutputPort(port);
		add.setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM);

		idx_t index = m_graph.addLayer(add);
		m_graph.connect({input1.m_index, input1.m_offset}, {index, 0});
		m_graph.connect({input2.m_index, input2.m_offset}, {index, 1});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addConcat(py::list inputs, SizeVector shape, const char *name)
	{
		std::vector<Tensor> tensors(py::len(inputs));
		std::vector<Port> ports(py::len(inputs));

		for (std::size_t i = 0; i < py::len(inputs); i++)
		{
			Tensor tensor = py::cast<Tensor>(inputs[i]);

			tensors[i] = tensor;
			ports[i] = Port(tensor.getShape());
		}

		auto concat = Builder::ConcatLayer(name).setAxis(1).setOutputPort(Port({shape}));
		concat.setInputPorts(ports);

		idx_t index = m_graph.addLayer(concat);

		for (std::size_t i = 0; i < py::len(inputs); i++)
			m_graph.connect({tensors[i].m_index, tensors[i].m_offset}, {index, i});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addFlatten(Tensor input, SizeVector shape, const char *name)
	{
		auto reshape = Builder::ReshapeLayer(name).setDims({0, -1});
		reshape.setInputPort(Port(input.getShape())).setOutputPort(Port(shape));

		idx_t index = m_graph.addLayer(reshape);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addLinear(Tensor input, SizeVector shape, std::size_t Wdata, std::size_t biasdata, const char *name)
	{
		SizeVector inshape = input.getShape();

		auto Wblob = make_shared_blob<float>(TensorDesc(Precision::FP32, {shape[1], inshape[1]}, Layout::NC));
		Wblob->allocate();

		void *WblobPtr = Wblob->buffer().as<void *>(), *Wptr = reinterpret_cast<void *>(Wdata);
		std::memcpy(WblobPtr, Wptr, Wblob->byteSize());

		auto linear = Builder::FullyConnectedLayer(name).setOutputNum(shape[1]);
		linear.setInputPort(Port(inshape)).setOutputPort(Port(shape));

		idx_t Windex = m_graph.addLayer(Builder::ConstLayer("W").setData(Wblob));
		idx_t index = m_graph.addLayer(linear);

		m_graph.connect({input.m_index, input.m_offset}, {index, 0});
		m_graph.connect({Windex}, {index, 1});

		if (biasdata != 0)
		{
			auto biasBlob = make_shared_blob<float>(TensorDesc(Precision::FP32, {shape[1]}, Layout::C));
			biasBlob->allocate();

			void *biasBlobPtr = biasBlob->buffer().as<void *>(), *biasptr = reinterpret_cast<void *>(biasdata);
			std::memcpy(biasBlobPtr, biasptr, biasBlob->byteSize());

			idx_t biasIndex = m_graph.addLayer(Builder::ConstLayer("bias").setData(biasBlob));
			m_graph.connect({biasIndex}, {index, 2});
		}

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	Tensor addSoftMax(Tensor input, const char *name)
	{
		idx_t index = m_graph.addLayer(Builder::SoftMaxLayer(name).setAxis(1).setPort(Port(input.getShape())));
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}

	std::vector<Tensor> addSplit(Tensor input, std::size_t axis, std::vector<SizeVector> shapes, const char *name)
	{
		assert(axis == 1);

		auto split = Builder::SplitLayer(name).setAxis(axis);
		split.setInputPort(Port(input.getShape()));

		std::vector<Port> ports(shapes.size());

		for (std::size_t i = 0; i < shapes.size(); i++)
			ports[i] = Port(shapes[i]);

		split.setOutputPorts(ports);

		idx_t index = m_graph.addLayer(split);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		std::vector<Tensor> tensors(shapes.size());

		for (std::size_t i = 0; i < shapes.size(); i++)
			tensors[i] = Tensor{&m_graph, index, i};

		return tensors;
	}

	Tensor addUpsample(Tensor input, int scale, const char *name)
	{
		auto resample = Builder::ResampleLayer(name).setResampleType("caffe.ResampleParameter.NEAREST");
		resample.setFactor(static_cast<float>(scale));

		auto shape = input.getShape();
		resample.setInputPort(Port(shape));

		for (std::size_t i = 2; i < shape.size(); i++)
			shape[i] *= scale;

		resample.setOutputPort(Port(shape));

		idx_t index = m_graph.addLayer(resample);
		m_graph.connect({input.m_index, input.m_offset}, {index});

		Tensor tensor = {&m_graph, index};
		return tensor;
	}
};


Graph *createNetwork(std::string name)
{
	return new Graph(name);
}


struct VINOEngine
{
	ExecutableNetwork m_engine;


	VINOEngine(std::size_t batchsize, const std::string &xmlpath, const std::string &binpath,
				   const std::string &backend)
	{
		CNNNetReader reader;

		reader.ReadNetwork(xmlpath);
		reader.ReadWeights(binpath);

		CNNNetwork net = reader.getNetwork();
		net.setBatchSize(batchsize);

		Core ie;

		if (backend == "CPU")
		{
#ifdef _WIN32
			const char *extlib = "cpu_extension_avx2.dll";
#else
			const char *extlib = "libcpu_extension_avx2.so";
#endif

			auto extension = make_so_pointer<::InferenceEngine::IExtension>(extlib);
			ie.AddExtension(extension, backend);
		}

		m_engine = ie.LoadNetwork(net, backend);
	}

	void infer(py::dict outputs, py::dict inputs)
	{
		InferRequest request = m_engine.CreateInferRequest();

		for (auto &info : m_engine.GetInputsInfo())
		{
			py::tuple input = inputs[info.first.c_str()];

			float *ptr = reinterpret_cast<float *>(py::cast<size_t>(input[0]));
			size_t size = py::cast<size_t>(input[1]);

			auto blob = make_shared_blob<float>(info.second->getTensorDesc(), ptr, size);
			request.SetBlob(info.first, blob);
		}

		request.Infer();

		for (auto &info : m_engine.GetOutputsInfo())
		{
			auto blob = request.GetBlob(info.first);
			void *ptr = blob->buffer().as<void *>();

			py::tuple output = outputs[info.first.c_str()];

			void *outptr = reinterpret_cast<void *>(py::cast<size_t>(output[0]));
			size_t size = py::cast<size_t>(output[1]);

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
		.def(py::init<std::size_t, const std::string&, const std::string&, const std::string&>())
		.def("infer", &VINOEngine::infer);
}
