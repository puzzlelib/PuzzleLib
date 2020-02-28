#ifndef DEVICE_H
#define DEVICE_H

#include "Common.h"


struct Device
{
	cl::Device m_device;
	cl::string m_name;
	cl::string m_version;
	size_t m_maxWorkGroupSize;
	size_t m_maxComputeUnits;


	Device() = default;
	Device(cl::Device device);
	std::string name() { return m_name; }
	std::string version() { return m_version; }
	size_t memBaseAddrAlign();
	size_t maxWorkGroupSize() { return m_maxWorkGroupSize; };
	size_t maxComputeUnits() { return m_maxComputeUnits; };
};


void initDevice(py::module &m);


#endif
