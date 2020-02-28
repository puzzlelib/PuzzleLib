#include <cassert>
#include <cuda_fp16.h>

#include "ReflectPad1D.h"


__forceinline__ __device__ void map1d(int insize, int outsize, int index, int lpad, int& inindex, int& outindex)
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


ReflectPad1D::ReflectPad1D(int lpad, int rpad) : m_lpad(lpad), m_rpad(rpad)
{

}

ReflectPad1D::ReflectPad1D(const void *serialData, size_t serialLength) : Plugin(serialData, serialLength)
{
	const char *buffer = static_cast<const char *>(serialData) + Plugin::getSerializationSize();

	read(buffer, m_lpad);
	read(buffer, m_rpad);
}

size_t ReflectPad1D::getSerializationSize()
{
	return Plugin::getSerializationSize() + sizeof(m_lpad) + sizeof(m_rpad);
}

void ReflectPad1D::serialize(void *serialData)
{
	Plugin::serialize(serialData);

	char *buffer = static_cast<char *>(serialData) + Plugin::getSerializationSize();
	write(buffer, m_lpad);
	write(buffer, m_rpad);
}

int ReflectPad1D::enqueue(int batchSize, const void * const *inputs, void **outputs, void */*workspace*/,
						  cudaStream_t stream)
{
	int maps = m_inshape.d[0], insize = m_inshape.d[1];
	int outsize = insize + m_lpad + m_rpad;

	dim3 block(32);
	dim3 grid(
		(outsize + block.x - 1) / block.x,
		maps,
		batchSize
	);

	if (m_datatype == nv::DataType::kFLOAT)
	{
		reflectpad1d<<<grid, block, 0, stream>>>(
			static_cast<float *>(outputs[0]), static_cast<const float *>(inputs[0]), insize, m_lpad, m_rpad
		);
	}
	else
	{
		reflectpad1d<<<grid, block, 0, stream>>>(
			static_cast<half *>(outputs[0]), static_cast<const half *>(inputs[0]), insize, m_lpad, m_rpad
		);
	}

	return 0;
}

nv::Dims ReflectPad1D::getOutputDimensions(int index, const nv::Dims *inputDims, int nbInputs)
{
	assert(nbInputs == 1 && index == 0);
	nv::Dims inshape = inputDims[0];

	nv::Dims outshape;
	outshape.nbDims = inshape.nbDims;

	outshape.type[0] = inshape.type[0];
	outshape.d[0] = inshape.d[0];

	outshape.type[1] = inshape.type[1];
	outshape.d[1] = inshape.d[1] + m_lpad + m_rpad;

	return outshape;
}

void ReflectPad1D::configureWithFormat(const nv::Dims *inputDims, int /*nbInputs*/, const nv::Dims *outputDims,
									   int /*nbOutputs*/, nv::DataType type, nv::PluginFormat format,
									   int /*maxBatchSize*/)
{
	m_inshape = inputDims[0];
	m_outshape = outputDims[0];

	assert((type == nv::DataType::kFLOAT || type == nv::DataType::kHALF) && format == nv::PluginFormat::kNCHW);
	m_datatype = type;
}

bool ReflectPad1D::supportsFormat(nv::DataType type, nv::PluginFormat format) const
{
	return (type == nv::DataType::kFLOAT || type == nv::DataType::kHALF) && format == nv::PluginFormat::kNCHW;
}

int ReflectPad1D::initialize()
{
	return 0;
}

void ReflectPad1D::terminate()
{

}

size_t ReflectPad1D::getWorkspaceSize(int /*maxBatchSize*/) const
{
	return 0;
}

int ReflectPad1D::getNbOutputs() const
{
	return 1;
}
