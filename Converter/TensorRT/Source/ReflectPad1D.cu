#include <cuda_fp16.h>
#include "Plugins.h"


__forceinline__ __device__ void map1d(int insize, int outsize, int index, int lpad, int &inindex, int &outindex)
{
	int inoffset = (blockIdx.y + blockIdx.z * gridDim.y) * insize;
	int outoffset = (blockIdx.y + blockIdx.z * gridDim.y) * outsize;

	int instart = max(0, -lpad);
	int outstart = max(0, lpad);

	int x = abs(index - lpad) - abs(index - (insize + lpad - 1)) - index + 2 * lpad + insize - 1 - outstart + instart;
	inindex = inoffset + x, outindex = outoffset + index;
}


template <typename Dtype>
__global__ void reflectpad1d(Dtype* outdata, const Dtype* indata, int insize, int lpad, int rpad)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outsize = insize + lpad + rpad;

	if (index < outsize)
	{
		int inindex = 0, outindex = 0;
		map1d(insize, outsize, index, lpad, inindex, outindex);

		outdata[outindex] = indata[inindex];
	}
}


struct ReflectPad1D : PuzzlePlugin
{
	int m_lpad, m_rpad;


	const char *getPluginType() const override { return PuzzlePluginCreator::reflectPad1DName; }
	const char *getPluginVersion() const override { return PuzzlePluginCreator::version; }

	ReflectPad1D(int lpad, int rpad) : m_lpad(lpad), m_rpad(rpad) {}

	ReflectPad1D(const void *serialData, size_t serialLength) : PuzzlePlugin(serialData, serialLength)
	{
		const char *buffer = static_cast<const char *>(serialData) + PuzzlePlugin::getSerializationSize();

		readValue(buffer, m_lpad);
		readValue(buffer, m_rpad);
	}

	size_t getSerializationSize() const override
	{
		return PuzzlePlugin::getSerializationSize() + sizeof(m_lpad) + sizeof(m_rpad);
	}

	void serialize(void *serialData) const override
	{
		PuzzlePlugin::serialize(serialData);
		char *buffer = static_cast<char *>(serialData) + PuzzlePlugin::getSerializationSize();

		writeValue(buffer, m_lpad);
		writeValue(buffer, m_rpad);
	}


	int getNbOutputs() const override { return 1; }
	nv::Dims getOutputDimensions(int index, const nv::Dims *inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1 && index == 0);

		auto inshape = inputs[0];
		auto outshape = nv::Dims2(inshape.d[0], inshape.d[1] + m_lpad + m_rpad);

		return outshape;
	}

	int initialize() override { return 0; }
	void terminate() override {}
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
	nv::IPluginV2Ext *clone() const override { return new ReflectPad1D(*this); }


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


	size_t getWorkspaceSize(int maxBatchSize) const override { (void)maxBatchSize; return 0; }

	int enqueue(int batchSize, const void * const *inputs, void **outputs, void *ws, cudaStream_t stream) override
	{
		(void)ws;

		int maps = m_inshape.d[0], insize = m_inshape.d[1];
		int outsize = insize + m_lpad + m_rpad;

		dim3 block(32);
		dim3 grid(
			(outsize + block.x - 1) / block.x, maps, batchSize
		);

		if (m_datatype == nv::DataType::kFLOAT)
		{
			reflectpad1d<float><<<grid, block, 0, stream>>>(
				static_cast<float *>(outputs[0]), static_cast<const float *>(inputs[0]), insize, m_lpad, m_rpad
			);
		}
		else
		{
			reflectpad1d<half><<<grid, block, 0, stream>>>(
				static_cast<half *>(outputs[0]), static_cast<const half *>(inputs[0]), insize, m_lpad, m_rpad
			);
		}

		return 0;
	}
};


static const nv::PluginField reflectPad1DFields[] = {
	{"pad", nullptr, nv::PluginFieldType::kDIMS, 1}
};

static const nv::PluginFieldCollection reflectPad1DFC = {1, reflectPad1DFields};


struct ReflectPad1DCreator : PuzzlePluginCreator
{
	const char *getPluginName() const override { return PuzzlePluginCreator::reflectPad1DName; }
	const nv::PluginFieldCollection *getFieldNames() override { return &reflectPad1DFC; }

	nv::IPluginV2 *createPlugin(const char *name, const nv::PluginFieldCollection *fc) override
	{
		(void)name;
		assert(fc->nbFields == 1);

		auto padding = fc->fields[0];
		assert(padding.type == nv::PluginFieldType::kDIMS);

		auto pad = *reinterpret_cast<const nv::Dims *>(padding.data);
		return new ReflectPad1D(pad.d[0], pad.d[1]);
	}

	nv::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
	{
		(void)name;
		return new ReflectPad1D(serialData, serialLength);
	}
};


REGISTER_TENSORRT_PLUGIN(ReflectPad1DCreator);
