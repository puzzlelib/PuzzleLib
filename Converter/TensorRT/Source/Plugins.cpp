#include "Plugins.h"


PuzzlePlugin::PuzzlePlugin() {}

PuzzlePlugin::PuzzlePlugin(const void *serialData, size_t serialLength)
{
	(void)serialLength;
	const char *buffer = static_cast<const char *>(serialData);

	readValue(buffer, m_inshape);
	readValue(buffer, m_outshape);
	readValue(buffer, m_datatype);
}

size_t PuzzlePlugin::getSerializationSize() const
{
	return sizeof(m_inshape) + sizeof(m_outshape) + sizeof(m_datatype);
}

void PuzzlePlugin::serialize(void *serialData) const
{
	char *buffer = static_cast<char *>(serialData);

	writeValue(buffer, m_inshape);
	writeValue(buffer, m_outshape);
	writeValue(buffer, m_datatype);
}

void PuzzlePlugin::setPluginNamespace(const char *pluginNamespace) { m_ns = std::string(pluginNamespace); }
const char *PuzzlePlugin::getPluginNamespace() const { return m_ns.c_str(); }


const char *PuzzlePluginCreator::getPluginVersion() const { return version; }
void PuzzlePluginCreator::setPluginNamespace(const char *pluginNamespace) { m_ns = std::string(pluginNamespace); }
const char *PuzzlePluginCreator::getPluginNamespace() const { return m_ns.c_str(); }


const char *PuzzlePluginCreator::version = "1";
const char *PuzzlePluginCreator::reflectPad1DName = "reflectpad1d";
const char *PuzzlePluginCreator::instNorm2DName = "instnorm2d";
