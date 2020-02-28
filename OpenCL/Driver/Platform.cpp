#include "Platform.h"


Platform::Platform(cl::Platform platform) :
	m_platform(platform)
{
	m_name = platform.getInfo<CL_PLATFORM_NAME>();
}


py::list Platform::getDevices(DeviceType type)
{
	std::vector<cl::Device> allDevices;
	m_platform.getDevices(type, &allDevices);

	py::list lst;
	for (auto device : allDevices)
		lst.append(Device(device));

	return lst;
}


py::list getPlatforms()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	py::list lst;
	for (auto platform : platforms)
		lst.append(Platform(platform));

	return lst;
}


void initPlatform(py::module &m)
{
	py::class_<Platform>(m, "Platform")
		.def_property_readonly("name", &Platform::name)
		.def("get_devices", &Platform::getDevices, py::arg("type"));

	m.def("get_platforms", &getPlatforms);
}
