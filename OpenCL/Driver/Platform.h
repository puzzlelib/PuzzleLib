#ifndef PLATFORM_H
#define PLATFORM_H

#include "Common.h"
#include "Device.h"


struct Platform
{
	cl::Platform m_platform;
	cl::string m_name;


	Platform(cl::Platform platform);
	std::string name() { return m_name; }
	py::list getDevices(DeviceType type);
};


py::list getPlatforms();


void initPlatform(py::module &m);


#endif
