#include <algorithm>

#include "Array.h"


static auto int8 = py::dtype("int8");
static auto uint8 = py::dtype("uint8");
static auto int16 = py::dtype("int16");
static auto uint16 = py::dtype("uint16");
static auto int32 = py::dtype("int32");
static auto uint32 = py::dtype("uint32");
static auto int64 = py::dtype("int64");
static auto uint64 = py::dtype("uint64");
static auto float32 = py::dtype("float32");


Array::Array(CommandQueue queue, py::object sh, py::object dtype, std::shared_ptr<Buffer> data, size_t offset,
	std::shared_ptr<MemoryPool> allocator) :
	m_queue(queue),
	m_offset(offset),
	m_size(1)
{
	auto shape = py::cast<py::tuple>(sh);

	m_shape = py::tuple(shape.size());
	m_strides = py::tuple(shape.size());

	m_dtype = py::dtype::from_args(dtype);
	m_itemsize = m_dtype.itemsize();

	for (ssize_t i = shape.size() - 1; i >= 0; i--)
	{
		auto dim = shape[i].cast<size_t>();
		if (dim == 0)
			throw std::runtime_error("Invalid dimension");

		m_shape[i] = dim;
		m_strides[i] = m_size * m_itemsize;

		m_size *= dim;
	}

	if (data == nullptr)
	{
		if (allocator == nullptr)
			m_buffer = std::make_shared<Buffer>(queue.m_context, m_size * m_itemsize);
		else
			m_buffer = allocate(allocator, m_size * m_itemsize);
	}
	else
		m_buffer = data;
}


void Array::set(py::array ary, bool async)
{
	if (static_cast<size_t>(ary.size()) != m_size)
		throw std::runtime_error("Invalid numpy size");

	if (!py::isinstance(ary.dtype(), m_dtype.get_type()))
		throw std::runtime_error("Invalid numpy dtype");

	m_queue.m_queue.enqueueWriteBuffer(m_buffer->m_buffer, !async, m_offset, nbytes(), ary.data());
}


py::object Array::get(bool async)
{
	std::vector<size_t> shape;
	shape.reserve(m_shape.size());

	for (size_t i = 0; i < ndim(); i++)
		shape.push_back(m_shape[i].cast<size_t>());

	bool isValue = false;
	if (shape.size() == 0)
	{
		isValue = true;
		shape.push_back(1);
	}
	
	auto ary = py::array(m_dtype, shape);
	m_queue.m_queue.enqueueReadBuffer(m_buffer->m_buffer, !async, m_offset, nbytes(), ary.mutable_data());

	if (isValue)
	{
		if (m_dtype.is(float32))
			return py::float_(ary);
		else
			return py::int_(ary);
	}
	else
		return ary;
}


template<typename T>
void fill(Array* ary, T value)
{
	T number = static_cast<T>(value);
	ary->m_queue.m_queue.enqueueFillBuffer(ary->m_buffer->m_buffer, number, ary->m_offset, ary->nbytes());
}


void Array::fill(py::object value)
{
	if (m_dtype.is(float32))
		::fill(this, value.cast<float>());
	else if (m_dtype.is(int64))
		::fill(this, value.cast<long long>());
	else if (m_dtype.is(int32))
		::fill(this, value.cast<int>());
	else if (m_dtype.is(uint8))
		::fill(this, value.cast<unsigned char>());
	else if (m_dtype.is(int8))
		::fill(this, value.cast<signed char>());
	else if (m_dtype.is(int16))
		::fill(this, value.cast<short>());
	else if (m_dtype.is(uint16))
		::fill(this, value.cast<unsigned short>());
	else if (m_dtype.is(uint64))
		::fill(this, value.cast<unsigned long long>());
	else if (m_dtype.is(uint32))
		::fill(this, value.cast<unsigned int>());
	else
		throw std::runtime_error("Invalid fill value");
}


Array* Array::reshape(py::args shape)
{
	if (shape.size() == 1 && py::isinstance<py::tuple>(shape[0]))
		shape = shape[0].cast<py::tuple>();
	
	py::tuple outshape(shape.size());

	size_t size = 1;
	ssize_t undefIndex = -1;
	
	for (size_t i = 0; i < shape.size(); i++)
	{
		ssize_t dim = shape[i].cast<ssize_t>();
		if (dim < 0)
		{
			if (undefIndex >= 0)
				throw std::runtime_error("Invalid shape");

			undefIndex = i;
		}
		else
			size *= dim;
		
		outshape[i] = dim;
	}

	if (undefIndex >= 0)
	{
		if (m_size % size != 0)
			throw std::runtime_error("Invalid shape");
		
		outshape[undefIndex] = m_size / size;
		size = m_size;
	}

	if (m_size != size)
		throw std::runtime_error("Invalid new shape");

	auto ary = new Array(m_queue, outshape, m_dtype, m_buffer, m_offset);
	return ary;
}


Array* Array::ravel()
{
	py::tuple outshape(1);
	outshape[0] = m_size;

	return new Array(m_queue, outshape, m_dtype, m_buffer, m_offset);
}


Array* Array::select(py::object index)
{
	ssize_t start = 0, stop = 0;
	ssize_t dim = m_shape[0].cast<ssize_t>();

	size_t ndim = 0;
	bool isIndex = false;

	if (py::isinstance<py::slice>(index))
	{
		auto slc = index.cast<py::slice>();

		py::object step = slc.attr("step");
		if (!py::isinstance<py::none>(step))
			throw std::runtime_error("Non-contiguous slice step");

		py::object objStart = slc.attr("start"); 
		py::object objStop = slc.attr("stop");
		
		start = py::isinstance<py::none>(objStart) ?  0 : objStart.cast<ssize_t>();
		if (start < 0) start = dim + start;
		start = std::min(start, dim);

		stop = py::isinstance<py::none>(objStop) ? dim : objStop.cast<ssize_t>();
		if (stop < 0) stop = dim + stop;
		stop = std::min(stop, dim);

		ndim = m_shape.size();
	}
	else if (py::isinstance<py::int_>(index))
	{
		ssize_t number = index.cast<ssize_t>();
		if (number < 0) number = dim + number;
		
		start = number;
		stop = number + 1;

		ndim = m_shape.size() - 1;
		isIndex = true;
	}
	else
		throw std::runtime_error("Unrecognized indexing argument");

	if (start >= stop || stop > dim || start > dim || stop < 0 || start < 0)
		throw std::runtime_error("Out of bounds slice");

	size_t offset = start * m_strides[0].cast<size_t>();

	py::tuple shape(ndim);

	for (size_t i = 0; i < ndim; i++)
		shape[i] = m_shape[i + isIndex];

	if (!isIndex)
		shape[0] = stop - start;

	return new Array(m_queue, shape, m_dtype, m_buffer, offset, nullptr);
}


py::str Array::toString()
{
	return py::str(get());
}


Array* toDevice(CommandQueue queue, py::array ary, std::shared_ptr<MemoryPool> allocator)
{
	py::tuple shape(ary.ndim());
	for (ssize_t i = 0; i < ary.ndim(); i++)
		shape[i] = ary.shape()[i];

	auto devAry = new Array(queue, shape, ary.dtype(), nullptr, 0, allocator);
	devAry->set(ary);

	return devAry;
}


Array* zeros(CommandQueue queue, py::object shape, py::object dtype, std::shared_ptr<MemoryPool> allocator)
{
	auto ary = new Array(queue, shape, dtype, nullptr, 0, allocator);
	queue.m_queue.enqueueFillBuffer(ary->m_buffer->m_buffer, static_cast<cl_uchar>(0), ary->m_offset, ary->nbytes());

	return ary;
}


Array* emptyLike(Array* ary)
{
	return new Array(ary->m_queue, ary->m_shape, ary->m_dtype, nullptr, 0, ary->m_buffer->m_allocator);
}


Array* zerosLike(Array* ary)
{
	return zeros(ary->m_queue, ary->m_shape, ary->m_dtype, ary->m_buffer->m_allocator);
}


py::tuple splay(Context* context, size_t n)
{
	auto device = context->m_device;

	size_t maxWorkItems = std::min<size_t>(128, device.m_maxWorkGroupSize);
	size_t minWorkItems = std::min<size_t>(32, maxWorkItems);

	size_t maxGroups = device.m_maxComputeUnits * 4 * 8;
	size_t groupCount = 0, workItemsPerGroup = 0;

	if (n < minWorkItems)
	{
		groupCount = 1;
		workItemsPerGroup = minWorkItems;
	}
	else if (n < maxGroups * minWorkItems)
	{
		groupCount = (n + minWorkItems - 1) / minWorkItems;
		workItemsPerGroup = minWorkItems;
	}
	else if (n < maxGroups * maxWorkItems)
	{
		groupCount = maxGroups;
		size_t grp = (n + minWorkItems - 1) / minWorkItems;
		workItemsPerGroup = ((grp + maxGroups - 1) / maxGroups) * minWorkItems;
	}
	else
	{
		groupCount = maxGroups;
		workItemsPerGroup = maxWorkItems;
	}

	py::tuple result(2);
	result[0] = groupCount * workItemsPerGroup;
	result[1] = workItemsPerGroup;

	return result;
}


void initArray(py::module &m)
{
	py::class_<Array>(m, "Array")
		.def(py::init<CommandQueue, py::object, py::object, std::shared_ptr<Buffer>, size_t,
			std::shared_ptr<MemoryPool>>(),
			py::arg("queue"), py::arg("shape"), py::arg("dtype"), py::arg("data").none() = py::none(),
			py::arg("offset") = 0, py::arg("allocator").none() = py::none())
		.def_property_readonly("data", &Array::data)
		.def_property_readonly("base_data", &Array::data)
		.def_property_readonly("shape", &Array::shape)
		.def_property_readonly("strides", &Array::strides)
		.def_property_readonly("offset", &Array::offset)
		.def_property_readonly("item_offset", &Array::itemOffset)
		.def_property_readonly("size", &Array::size)
		.def_property_readonly("nbytes", &Array::nbytes)
		.def_property_readonly("ndim", &Array::ndim)
		.def_property_readonly("dtype", &Array::dtype)
		.def_property_readonly("int_ptr", &Array::intPtr)
		.def("set", &Array::set, py::arg("ary"), py::arg("async") = false)
		.def("get", &Array::get, py::arg("async") = false)
		.def("fill", &Array::fill, py::arg("value"))
		.def("reshape", &Array::reshape)
		.def("ravel", &Array::ravel)
		.def("__getitem__", &Array::select)
		.def("__str__", &Array::toString);

	m.def("to_device", &toDevice, py::arg("queue"), py::arg("ary"), py::arg("allocator").none() = py::none());
	m.attr("empty") = m.attr("Array");
	m.def("zeros", &zeros, py::arg("queue"), py::arg("shape"), py::arg("dtype"),
		py::arg("allocator").none() = py::none());
	m.def("empty_like", &emptyLike, py::arg("ary").none(false));
	m.def("zeros_like", &zerosLike, py::arg("ary").none(false));
	m.def("splay", &splay, py::arg("context").none(false), py::arg("n"));
}
