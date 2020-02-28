#ifndef ARRAY_H
#define ARRAY_H

#include <vector>

#include <pybind11/numpy.h>

#include "Buffer.h"
#include "MemoryPool.h"
#include "Queue.h"


#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif
struct Array
{
	CommandQueue m_queue;
	std::shared_ptr<Buffer> m_buffer;

	py::tuple m_shape;
	py::tuple m_strides;

	size_t m_offset;
	size_t m_size;
	size_t m_itemsize;
	py::dtype m_dtype;


	Array(CommandQueue queue, py::object shape, py::object dtype,
		std::shared_ptr<Buffer> data=std::shared_ptr<Buffer>(), size_t offset=0,
		std::shared_ptr<MemoryPool> allocator=std::shared_ptr<MemoryPool>());
	std::shared_ptr<Buffer> data() { return m_buffer; }
	py::tuple shape() { return m_shape; }
	py::tuple strides() { return m_strides; }
	size_t offset() { return m_offset; }
	size_t itemOffset() { return m_offset / m_itemsize; }
	size_t size() { return m_size; }
	size_t nbytes() { return m_size * m_itemsize; }
	size_t ndim() { return m_shape.size(); }
	py::dtype dtype() { return m_dtype; }
	Array* reshape(py::args shape);
	Array* ravel();
	Array* select(py::object index);
	void set(py::array ary, bool async=false);
	py::object get(bool async=false);
	void fill(py::object value);
	size_t intPtr() { return m_buffer->intPtr(); }
	py::str toString();
};
#ifdef __GNUC__
#pragma GCC visibility pop
#endif


Array* toDevice(CommandQueue queue, py::array ary, std::shared_ptr<MemoryPool> allocator=std::shared_ptr<MemoryPool>());
Array* zeros(CommandQueue queue, py::object shape, py::object dtype,
	std::shared_ptr<MemoryPool> allocator=std::shared_ptr<MemoryPool>());
Array* emptyLike(Array* ary);
Array* zerosLike(Array* ary);

py::tuple splay(Context* context, size_t n);


void initArray(py::module &m);


#endif
