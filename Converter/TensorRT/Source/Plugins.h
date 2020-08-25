#pragma once

#include <cassert>
#include <cstring>

#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef __GNUC__
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <NvInfer.h>
namespace nv = nvinfer1;

#include <NvInferPlugin.h>
#include <NvCaffeParser.h>

#ifdef __GNUC__
	#pragma GCC diagnostic pop
#endif


struct PuzzlePlugin : nv::IPluginV2IOExt
{
	std::string m_ns;
	nv::Dims m_inshape, m_outshape;
	nv::DataType m_datatype;


	PuzzlePlugin();
	PuzzlePlugin(const void *serialData, size_t serialLength);

	size_t getSerializationSize() const override;
	void serialize(void *serialData) const override;

	void setPluginNamespace(const char *pluginNamespace) override;
	const char *getPluginNamespace() const override;


	template<typename T>
	static void writeValue(char *&buffer, const T &val)
	{
		*reinterpret_cast<T *>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T>
	static void writeVector(char *&buffer, const std::vector<T> &vector)
	{
		size_t size = vector.size();
		writeValue(buffer, size);

		size_t nbytes = size * sizeof(T);
		std::memcpy(buffer, &vector[0], nbytes);

		buffer += nbytes;
	}

	template<typename T>
	static void readValue(const char *&buffer, T &val)
	{
		val = *reinterpret_cast<const T *>(buffer);
		buffer += sizeof(T);
	}

	template<typename T>
	static void readVector(const char *&buffer, std::vector<T> &vector)
	{
		size_t size;
		readValue(buffer, size);

		vector.resize(size);
		size_t nbytes = size * sizeof(T);

		std::memcpy(&vector[0], buffer, nbytes);
		buffer += nbytes;
	}

	template<typename T>
	static size_t vectorNBytes(const std::vector<T> &vector)
	{
		return sizeof(vector.size()) + vector.size() * sizeof(T);
	}
};


struct PuzzlePluginCreator : nv::IPluginCreator
{
	std::string m_ns;


	const char *getPluginVersion() const override;
	void setPluginNamespace(const char *pluginNamespace) override;
	const char *getPluginNamespace() const override;


	static const char *version, *reflectPad1DName, *instNorm2DName;
};


template<typename T>
struct CudaBuffer
{
	T *m_data;
	size_t m_length;


	CudaBuffer(size_t length)
	{
		m_length = length;

		cudaError_t status = cudaMalloc(&m_data, length * sizeof(T));
		assert(status == cudaSuccess);
	}

	~CudaBuffer()
	{
		if (m_data != nullptr)
			cudaFree(m_data);
	}

	cudaError_t set(const std::vector<T> &data, size_t offset = 0)
	{
		assert(m_length >= offset + data.size());
		return cudaMemcpy(m_data + offset, &data[0], data.size() * sizeof(T), cudaMemcpyHostToDevice);
	}
};
