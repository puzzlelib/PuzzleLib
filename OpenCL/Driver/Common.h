#ifndef COMMON_H
#define COMMON_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <pybind11/pybind11.h>
namespace py = pybind11;


enum DeviceType
{
	CPU = CL_DEVICE_TYPE_CPU,
	GPU = CL_DEVICE_TYPE_GPU,
	ALL = CL_DEVICE_TYPE_ALL
};


void initCommon(py::module &m);


#endif
