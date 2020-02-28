#pragma once

#include <string>
#include <vector>

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


struct Plugin : nv::IPluginExt
{
	nv::Dims m_inshape;
	nv::Dims m_outshape;
	nv::DataType m_datatype;


	Plugin();
	Plugin(const void *serialData, size_t /*serialLength*/);

	size_t getSerializationSize() override;
	void serialize(void *serialData) override;

	template<typename T>
	void write(char *& buffer, const T& val)
	{
		*reinterpret_cast<T *>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T>
	void read(const char *& buffer, T& val)
	{
		val = *reinterpret_cast<const T *>(buffer);
		buffer += sizeof(T);
	}
};


struct IPluginFactory : nv::IPluginFactory
{
	std::vector<Plugin *> m_plugins;


	virtual ~IPluginFactory();

	nv::IPluginExt *createPRelu(const float *data, size_t length);
	nv::IPluginExt *createPRelu(const void *serialData, size_t serialLength);

	nv::IPluginExt *createReflectPad1D(int lpad, int rpad);
	nv::IPluginExt *createReflectPad1D(const void *serialData, size_t serialLength);
};


struct PluginFactory : IPluginFactory
{
	static const std::string prelu;
	static const std::string reflectpad;


	nv::IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
};


struct CaffePluginFactory : IPluginFactory, nvcaffeparser1::IPluginFactory
{
	bool isPlugin(const char *layerName) override;
	nvinfer1::IPlugin *createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights) override;
	nv::IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override;
};
