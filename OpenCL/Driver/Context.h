#ifndef CONTEXT_H
#define CONTEXT_H

#include "Common.h"
#include "Device.h"


struct Context
{
	cl::Context m_context;
	Device m_device;


	Context(py::list devices);
};


void initContext(py::module &m);


#endif
