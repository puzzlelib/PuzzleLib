#include <algorithm>
#include <string>
#include <cassert>

#include "PRelu.h"
#include "ReflectPad1D.h"


Plugin::Plugin() {}

Plugin::Plugin(const void *serialData, size_t /*serialLength*/)
{
	const char *buffer = static_cast<const char *>(serialData);

	read(buffer, m_inshape);
	read(buffer, m_outshape);
	read(buffer, m_datatype);
}

size_t Plugin::getSerializationSize()
{
	return sizeof(m_inshape) + sizeof(m_outshape) + sizeof(m_datatype);
}

void Plugin::serialize(void *serialData)
{
	char *buffer = static_cast<char *>(serialData);

	write(buffer, m_inshape);
	write(buffer, m_outshape);
	write(buffer, m_datatype);
}


inline static bool endsWith(const std::string& value, const std::string& ending)
{
	if (ending.size() > value.size())
		return false;

	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


const std::string PluginFactory::prelu = ":prelu";
const std::string PluginFactory::reflectpad = ":reflectpad";


IPluginFactory::~IPluginFactory()
{
	for (auto plugin : m_plugins)
		delete plugin;
}

nv::IPluginExt *IPluginFactory::createPRelu(const float *data, size_t length)
{
	Plugin *plugin = new PRelu(data, length);
	m_plugins.push_back(plugin);

	return plugin;
}

nv::IPluginExt *IPluginFactory::createPRelu(const void *serialData, size_t serialLength)
{
	Plugin *plugin = new PRelu(serialData, serialLength);
	m_plugins.push_back(plugin);

	return plugin;
}

nv::IPluginExt *IPluginFactory::createReflectPad1D(int lpad, int rpad)
{
	Plugin *plugin = new ReflectPad1D(lpad, rpad);
	m_plugins.push_back(plugin);

	return plugin;
}

nv::IPluginExt *IPluginFactory::createReflectPad1D(const void *serialData, size_t serialLength)
{
	Plugin *plugin = new ReflectPad1D(serialData, serialLength);
	m_plugins.push_back(plugin);

	return plugin;
}

nv::IPlugin *PluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength)
{
	auto internalName = std::string(layerName);

	if (endsWith(internalName, prelu))
		return createPRelu(serialData, serialLength);

	else if (endsWith(internalName, reflectpad))
		return createReflectPad1D(serialData, serialLength);

	assert(false);
	return nullptr;
}


bool CaffePluginFactory::isPlugin(const char *layerName)
{
	auto tolower = [](char ch)
	{
		return static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
	};

	std::string strname = layerName;
	std::transform(strname.begin(), strname.end(), strname.begin(), tolower);

	return(strname.find("prelu") != std::string::npos);
}

nvinfer1::IPlugin *CaffePluginFactory::createPlugin(const char * /*layerName*/, const nvinfer1::Weights *weights,
													int nbWeights)
{
	assert(nbWeights == 1);
	assert(weights[0].type == nv::DataType::kFLOAT);

	return createPRelu(static_cast<const float *>(weights[0].values), weights[0].count);
}

nv::IPlugin *CaffePluginFactory::createPlugin(const char * /*layerName*/, const void *serialData, size_t serialLength)
{
	return createPRelu(serialData, serialLength);
}
