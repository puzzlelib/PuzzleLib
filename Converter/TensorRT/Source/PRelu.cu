#include <cassert>
#include <cuda_fp16.h>

#include "PRelu.h"


__global__ void prelu(int n, int maps, int dim, const float *indata, float *outdata, const float *slopedata)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		int c = (i / dim) % maps;
		outdata[i] = indata[i] > 0 ? indata[i] : indata[i] * slopedata[c];
	}
}


__global__ void preluHalf(int n, int maps, int dim, const half *indata, half *outdata, const float *slopedata)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		int c = (i / dim) % maps;

		float value = __half2float(indata[i]);
		outdata[i] = __float2half(value > 0 ? value : value * slopedata[c]);
	}
}


PRelu::PRelu(const float *scales, size_t length)
{
	m_scales.resize(length);
	memcpy(&m_scales[0], scales, length * sizeof(float));
}

PRelu::PRelu(const void *serialData, size_t serialLength) : Plugin(serialData, serialLength)
{
	size_t headerSize = Plugin::getSerializationSize();
	const char *buffer = static_cast<const char *>(serialData) + headerSize;

	size_t length = serialLength - headerSize;

	m_scales.resize(length / sizeof(float));
	memcpy(&m_scales[0], buffer, length);
}

size_t PRelu::getSerializationSize()
{
	return Plugin::getSerializationSize() + m_scales.size() * sizeof(float);
}

void PRelu::serialize(void *serialData)
{
	Plugin::serialize(serialData);

	size_t headerSize = Plugin::getSerializationSize();
	char *buffer = static_cast<char *>(serialData) + headerSize;

	memcpy(buffer, &m_scales[0], m_scales.size() * sizeof(float));
}

int PRelu::enqueue(int batchSize, const void * const *inputs, void **outputs, void */*workspace*/, cudaStream_t stream)
{
	int maps = m_inshape.d[0];
	int dim = m_inshape.d[1] * m_inshape.d[2];

	int n = batchSize * maps * dim;

	dim3 block(512);
	dim3 grid((n + block.x - 1) / block.x, 1, 1);

	if (m_datatype == nv::DataType::kFLOAT)
	{
		prelu<<<grid, block, 0, stream>>>(
			n, maps, dim, static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]), m_gpuscales
		);
	}
	else
	{
		preluHalf<<<grid, block, 0, stream>>>(
			n, maps, dim, static_cast<const half *>(inputs[0]), static_cast<half *>(outputs[0]), m_gpuscales
		);
	}

	return 0;
}

nv::Dims PRelu::getOutputDimensions(int /*index*/, const nv::Dims *inputDims, int /*nbInputs*/)
{
	return *inputDims;
}

void PRelu::configureWithFormat(const nv::Dims *inputDims, int /*nbInputs*/, const nv::Dims *outputDims,
								int /*nbOutputs*/, nv::DataType type, nv::PluginFormat format, int /*maxBatchSize*/)
{
	m_inshape = inputDims[0];
	m_outshape = outputDims[0];

	assert((type == nv::DataType::kFLOAT || type == nv::DataType::kHALF) && format == nv::PluginFormat::kNCHW);
	m_datatype = type;
}

bool PRelu::supportsFormat(nv::DataType type, nv::PluginFormat format) const
{
	return (type == nv::DataType::kFLOAT || type == nv::DataType::kHALF) && format == nv::PluginFormat::kNCHW;
}

int PRelu::initialize()
{
	m_gpulength = m_scales.size() * sizeof(float);

	cudaError_t status = cudaSuccess;
	status = cudaMalloc(&m_gpuscales, m_gpulength);

	if (status != cudaSuccess)
		return status;

	status = cudaMemcpy(m_gpuscales, &m_scales[0], m_gpulength, cudaMemcpyHostToDevice);
	return status;
}

void PRelu::terminate()
{
	if (m_gpuscales != nullptr)
	{
		cudaFree(m_gpuscales);

		m_gpuscales = nullptr;
		m_gpulength = 0;
	}
}

size_t PRelu::getWorkspaceSize(int /*maxBatchSize*/) const
{
	return 0;
}

int PRelu::getNbOutputs() const
{
	return 1;
}
