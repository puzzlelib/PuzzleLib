#ifndef PROGRAM_H
#define PROGRAM_H

#include <map>

#include "Common.h"
#include "Context.h"
#include "Queue.h"


enum class ArgType
{
	unknown,
	char_,
	uchar,
	short_,
	ushort,
	int_,
	uint,
	long_,
	ulong,
	float_,
	pointer
};


struct KernelInfo
{
	cl::Kernel m_kernel;
	std::vector<ArgType> m_args;
};


struct Kernel
{
	cl::Kernel m_kernel;
	std::shared_ptr<KernelInfo> m_info;
	cl::string m_name;


	Kernel(const std::string& name, std::shared_ptr<KernelInfo> info);
	std::string name() { return m_name; }
	size_t nargs() { return m_info->m_args.size(); }
	void call(CommandQueue* queue, py::tuple grid, py::tuple block, py::args args);
};


struct Program
{
	Context m_context;
	cl::string m_source;
	cl::Program m_program;
	std::map<std::string, std::shared_ptr<KernelInfo>> m_kernels;


	Program(Context context, const std::string& source);
	Program* build();
	std::vector<std::string> allKernels();
	Kernel getKernel(const std::string& name);
};


void initProgram(py::module &m);


#endif
