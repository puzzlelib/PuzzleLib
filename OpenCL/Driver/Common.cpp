#include <sstream>

#include "Common.h"


void initCommon(py::module &m)
{
	py::enum_<DeviceType>(m, "device_type")
		.value("CPU", DeviceType::CPU)
		.value("GPU", DeviceType::GPU)
		.value("ALL", DeviceType::ALL);

	py::register_exception_translator([](std::exception_ptr p)
	{
		try
		{
			if (p) std::rethrow_exception(p);
		}
		catch (const cl::Error& e)
		{
			std::stringstream buffer;
			buffer << e.what() << " (Error: " << e.err() << ")";

			PyErr_SetString(PyExc_RuntimeError, buffer.str().c_str());
		}
	});
}
