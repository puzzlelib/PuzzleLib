#include "Context.h"


Context::Context(py::list devices)
{
	cl::vector<cl::Device> allDevices;
	for (auto obj : devices)
	{
		auto dev = obj.cast<Device>();
		allDevices.push_back(dev.m_device);
	}

	m_context = cl::Context(allDevices);
	m_device = allDevices[0];
}


void initContext(py::module &m)
{
	py::class_<Context>(m, "Context")
		.def(py::init<py::list>(), py::arg("devices"));
}
