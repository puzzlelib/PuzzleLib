#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Program.h"
#include "Array.h"
#include "Buffer.h"


Program::Program(Context context, const std::string& source) :
	m_context(context),
	m_source(source)
{
	m_program = cl::Program(m_context.m_context, m_source);
}


Program* Program::build()
{
	try
	{
		m_program.build();
	}
	catch (cl::BuildError&)
	{
		auto info = m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
		throw std::runtime_error(info[0].second.c_str());
	}

	m_source = cl::string();

	cl::vector<cl::Kernel> kernels;
	m_program.createKernels(&kernels);
	
	for (auto kernel : kernels)
	{
		std::string name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
		size_t nargs = kernel.getInfo<CL_KERNEL_NUM_ARGS>();

		KernelInfo info;
		info.m_kernel = kernel;
		info.m_args.reserve(nargs);

		for (unsigned i = 0; i < nargs; i++)
		{
			auto arg = kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(i);
			ArgType type = ArgType::unknown;

			if (arg.back() == '*')
				type = ArgType::pointer;
			else if (arg == "float")
				type = ArgType::float_;
			else if (arg == "int")
				type = ArgType::int_;
			else if (arg == "long")
				type = ArgType::long_;
			else if (arg == "uint")
				type = ArgType::uint;
			else if (arg == "ulong")
				type = ArgType::ulong;
			else if (arg == "char")
				type = ArgType::char_;
			else if (arg == "uchar")
				type = ArgType::uchar;
			else if (arg == "short")
				type = ArgType::short_;
			else if (arg == "ushort")
				type = ArgType::ushort;

			info.m_args.push_back(type);
		}

		m_kernels[name] = std::make_shared<KernelInfo>(info);
	}

	return this;
}


std::vector<std::string> Program::allKernels()
{
	std::vector<std::string> names;
	names.reserve(m_kernels.size());

	for (auto it = m_kernels.begin(); it != m_kernels.end(); ++it)
		names.push_back(it->first);

	return names;
}


Kernel Program::getKernel(const std::string& name)
{
	auto r = m_kernels.find(name);
	if (r == m_kernels.end())
		throw std::runtime_error("Kernel is not found");

	return Kernel(name, r->second);
}


Kernel::Kernel(const std::string& name, std::shared_ptr<KernelInfo> info) :
	m_kernel(info->m_kernel),
	m_info(info),
	m_name(name)
{

}


void Kernel::call(CommandQueue* queue, py::tuple grid, py::tuple block, py::args args)
{
	if (args.size() != m_info->m_args.size())
		throw std::runtime_error("Invalid number of kernel arguments");

	if (grid.size() != 3)
		throw std::runtime_error("Invalid global group size");

	if (block.size() != 3)
		throw std::runtime_error("Invalid local group shape");

	for (unsigned i = 0; i < args.size(); i++)
	{
		py::object obj = args[i];

		if (py::isinstance<Array>(obj))
		{
			auto ary = py::cast<Array*>(obj);
			m_kernel.setArg(i, ary->m_buffer->m_buffer);
		}
		else if (py::isinstance<Buffer>(obj))
		{
			auto buf = py::cast<Buffer*>(obj);
			m_kernel.setArg(i, buf->m_buffer);
		}
		else
		{
			ArgType type = m_info->m_args[i];

			switch (type)
			{
				case ArgType::float_:
					m_kernel.setArg(i, obj.cast<float>());
					break;

				case ArgType::int_:
					m_kernel.setArg(i, obj.cast<int>());
					break;

				case ArgType::long_:
					m_kernel.setArg(i, obj.cast<long long>());
					break;

				case ArgType::uint:
					m_kernel.setArg(i, obj.cast<unsigned>());
					break;

				case ArgType::ulong:
					m_kernel.setArg(i, obj.cast<unsigned long long>());
					break;

				case ArgType::char_:
					m_kernel.setArg(i, obj.cast<signed char>());
					break;

				case ArgType::uchar:
					m_kernel.setArg(i, obj.cast<unsigned char>());
					break;

				case ArgType::short_:
					m_kernel.setArg(i, obj.cast<short>());
					break;

				case ArgType::ushort:
					m_kernel.setArg(i, obj.cast<unsigned short>());
					break;

				default:
					throw std::runtime_error("Unrecognized scalar argument type #" + std::to_string(i + 1));
			}
		}
	}

	cl::NDRange global(grid[0].cast<size_t>(), grid[1].cast<size_t>(), grid[2].cast<size_t>());
	cl::NDRange local(block[0].cast<size_t>(), block[1].cast<size_t>(), block[2].cast<size_t>());

	queue->m_queue.enqueueNDRangeKernel(m_kernel, cl::NDRange(0, 0, 0), global, local);
}


void initProgram(py::module &m)
{
	py::class_<Program>(m, "Program")
		.def(py::init<Context, const std::string&>(), py::arg("context"), py::arg("code"))
		.def_property_readonly("all_kernels", &Program::allKernels)
		.def("build", &Program::build)
		.def("get_kernel", &Program::getKernel, py::arg("name"))
		.def("__getattr__", &Program::getKernel, py::arg("name"));

	py::class_<Kernel>(m, "Kernel")
		.def_property_readonly("function_name", &Kernel::name)
		.def_property_readonly("num_args", &Kernel::nargs)
		.def("__call__", &Kernel::call, py::arg("queue").none(false), py::arg("grid"), py::arg("block"));
}
