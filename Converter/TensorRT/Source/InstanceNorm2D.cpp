#include <memory>
#include <cudnn.h>

#include "Plugins.h"


inline static cudnnDataType_t trtToCuDnnType(nv::DataType dtype)
{
	switch (dtype)
	{
		case nv::DataType::kFLOAT: return CUDNN_DATA_FLOAT;
		case nv::DataType::kHALF:  return CUDNN_DATA_HALF;

		default: assert(false); return CUDNN_DATA_FLOAT;
	}
}


struct InstanceNorm2D : PuzzlePlugin
{
	std::shared_ptr<std::vector<float>> m_scale, m_bias;
	float m_epsilon;

	std::shared_ptr<CudaBuffer<float>> m_gpuscale, m_gpubias;
	mutable int m_maxBatchSize;

	cudnnTensorDescriptor_t m_dataDesc = nullptr, m_outDesc = nullptr, m_scaleDesc = nullptr;
	cudnnHandle_t m_cudnn;


	const char *getPluginType() const override { return PuzzlePluginCreator::instNorm2DName; }
	const char *getPluginVersion() const override { return PuzzlePluginCreator::version; }

	InstanceNorm2D(const float *scale, const float *bias, size_t length, float epsilon) : m_epsilon(epsilon)
	{
		m_scale = std::make_shared<std::vector<float>>(scale, scale + length);
		m_bias = std::make_shared<std::vector<float>>(bias, bias + length);
	}

	InstanceNorm2D(const void *serialData, size_t serialLength) : PuzzlePlugin(serialData, serialLength)
	{
		const char *buffer = static_cast<const char *>(serialData) + PuzzlePlugin::getSerializationSize();

		readValue(buffer, m_epsilon);
		readValue(buffer, m_maxBatchSize);

		m_scale = std::make_shared<std::vector<float>>();
		readVector(buffer, *m_scale);

		m_bias = std::make_shared<std::vector<float>>();
		readVector(buffer, *m_bias);
	}

	size_t getSerializationSize() const override
	{
		size_t nbytes = PuzzlePlugin::getSerializationSize() + sizeof(m_epsilon) + sizeof(m_maxBatchSize);
		return nbytes + vectorNBytes(*m_scale) + vectorNBytes(*m_bias);
	}

	void serialize(void *serialData) const override
	{
		PuzzlePlugin::serialize(serialData);
		char *buffer = static_cast<char *>(serialData) + PuzzlePlugin::getSerializationSize();

		writeValue(buffer, m_epsilon);
		writeValue(buffer, m_maxBatchSize);

		writeVector(buffer, *m_scale);
		writeVector(buffer, *m_bias);
	}


	int getNbOutputs() const override { return 1; }
	nv::Dims getOutputDimensions(int index, const nv::Dims *inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1 && index == 0);
		return *inputs;
	}

	int initialize() override
	{
		m_gpuscale = std::make_shared<CudaBuffer<float>>(m_scale->size() * m_maxBatchSize);
		m_gpubias = std::make_shared<CudaBuffer<float>>(m_bias->size() * m_maxBatchSize);

		for (int i = 0; i < m_maxBatchSize; i += 1)
		{
			m_gpuscale->set(*m_scale, i * m_scale->size());
			m_gpubias->set(*m_bias, i * m_bias->size());
		}

		cudnnStatus_t status = cudnnCreateTensorDescriptor(&m_dataDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			return status;

		status = cudnnCreateTensorDescriptor(&m_outDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			return status;

		status = cudnnCreateTensorDescriptor(&m_scaleDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			return status;

		return 0;
	}

	void terminate() override
	{
		m_gpuscale.reset();
		m_gpubias.reset();

		if (m_dataDesc != nullptr)
		{
			cudnnDestroyTensorDescriptor(m_dataDesc);
			m_dataDesc = nullptr;
		}

		if (m_outDesc != nullptr)
		{
			cudnnDestroyTensorDescriptor(m_outDesc);
			m_outDesc = nullptr;
		}

		if (m_scaleDesc != nullptr)
		{
			cudnnDestroyTensorDescriptor(m_scaleDesc);
			m_scaleDesc = nullptr;
		}
	}

	void destroy() override { delete this; }


	nv::DataType getOutputDataType(int index, const nv::DataType *inputTypes, int nbInputs) const override
	{
		(void)nbInputs;

		assert(index == 0);
		assert(inputTypes[0] == nv::DataType::kFLOAT || inputTypes[0] == nv::DataType::kHALF);

		return inputTypes[0];
	}

	bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override
	{
		(void)outputIndex, (void)inputIsBroadcasted, (void)nbInputs;
		return false;
	}

	bool canBroadcastInputAcrossBatch(int inputIndex) const override { (void)inputIndex; return false; }
	nv::IPluginV2Ext *clone() const override { return new InstanceNorm2D(*this); }


	bool supportsFormatCombination(int pos, const nv::PluginTensorDesc *inOut,
								   int nbInputs, int nbOutputs) const override
	{
		assert(nbInputs == 1 && nbOutputs == 1);
		auto desc = inOut[pos];

		if (pos == 0)
			return (desc.type == nv::DataType::kHALF || desc.type == nv::DataType::kFLOAT) &&
					desc.format == nv::TensorFormat::kLINEAR;

		return desc.format == inOut[0].format && desc.type == inOut[0].type;
	}

	void configurePlugin(const nv::PluginTensorDesc *in, int nbInputs,
						 const nv::PluginTensorDesc *out, int nbOutputs) override
	{
		(void)nbInputs, (void)nbOutputs;
		m_inshape = in[0].dims, m_outshape = out[0].dims, m_datatype = in[0].type;
	}


	void attachToContext(cudnnContext *cudnn, cublasContext *cublas, nv::IGpuAllocator *allocator) override
	{
		(void)cublas, (void)allocator;
		m_cudnn = cudnn;
	}

	void detachFromContext() override { m_cudnn = nullptr; }

	size_t getWorkspaceSize(int maxBatchSize) const override
	{
		m_maxBatchSize = maxBatchSize;
		return 0;
	}

	int enqueue(int batchSize, const void * const *inputs, void **outputs, void *ws, cudaStream_t stream) override
	{
		(void)ws;
		cudnnStatus_t status = cudnnSetStream(m_cudnn, stream);

		if (status != CUDNN_STATUS_SUCCESS)
			return status;

		cudnnDataType_t dtype = trtToCuDnnType(m_datatype);
		int maps = m_inshape.d[0], height = m_inshape.d[1], width = m_inshape.d[2];

		cudnnSetTensor4dDescriptor(m_scaleDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, batchSize * maps, 1, 1);
		cudnnSetTensor4dDescriptor(m_dataDesc, CUDNN_TENSOR_NCHW, dtype, 1, batchSize * maps, height, width);
		cudnnSetTensor4dDescriptor(m_outDesc, CUDNN_TENSOR_NCHW, dtype, 1, batchSize * maps, height, width);

		float alpha = 1.0f, beta = 0.0f;

		status = cudnnBatchNormalizationForwardTraining(
			m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, m_dataDesc, inputs[0], m_outDesc, outputs[0],
			m_scaleDesc, m_gpuscale->m_data, m_gpubias->m_data, 1.0, nullptr, nullptr, m_epsilon, nullptr, nullptr
		);

		return status;
	}
};


static const nv::PluginField instNorm2DFields[] = {
	{"scale", nullptr, nv::PluginFieldType::kFLOAT32},
	{"bias", nullptr, nv::PluginFieldType::kFLOAT32}
};

static const nv::PluginFieldCollection instNorm2DFC = {2, instNorm2DFields};


struct InstanceNorm2DCreator : PuzzlePluginCreator
{
	const char *getPluginName() const override { return PuzzlePluginCreator::instNorm2DName; }
	const nv::PluginFieldCollection *getFieldNames() override { return &instNorm2DFC; }

	nv::IPluginV2 *createPlugin(const char *name, const nv::PluginFieldCollection *fc) override
	{
		(void)name;
		assert(fc->nbFields == 3);

		auto scale = fc->fields[0], bias = fc->fields[1], epsilon = fc->fields[2];
		assert(scale.type == bias.type && bias.type == epsilon.type && scale.type == nv::PluginFieldType::kFLOAT32);

		return new InstanceNorm2D(
			reinterpret_cast<const float *>(scale.data), reinterpret_cast<const float *>(bias.data), scale.length,
			*reinterpret_cast<const float *>(epsilon.data)
		);
	}

	nv::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
	{
		(void)name;
		return new InstanceNorm2D(serialData, serialLength);
	}
};


REGISTER_TENSORRT_PLUGIN(InstanceNorm2DCreator);
