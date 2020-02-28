#include "Device.h"


Device::Device(cl::Device device) :
	m_device(device)
{
	m_name = device.getInfo<CL_DEVICE_NAME>();
	m_version = device.getInfo<CL_DEVICE_VERSION>();

	m_maxWorkGroupSize = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	m_maxComputeUnits = m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
}


size_t Device::memBaseAddrAlign()
{
	return m_device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
}


void initDevice(py::module &m)
{
	py::class_<Device>(m, "Device")
		.def_property_readonly("name", &Device::name)
		.def_property_readonly("version", &Device::version)
		.def_property_readonly("mem_base_addr_align", &Device::memBaseAddrAlign)
		.def_property_readonly("max_workgroup_size", &Device::maxWorkGroupSize)
		.def_property_readonly("max_compute_units", &Device::maxComputeUnits);
}
