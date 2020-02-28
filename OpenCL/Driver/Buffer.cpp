#include <array>

#include "Buffer.h"
#include "MemoryPool.h"
#include "Queue.h"


Buffer::Buffer(Context context, size_t size) 
{
	m_size = size;
	m_buffer = cl::Buffer(context.m_context, CL_MEM_READ_WRITE, size);
}


Buffer::Buffer(Context context, size_t size, std::shared_ptr<MemoryPool> allocator) :
	m_allocator(allocator)
{
	m_size = size;
	m_buffer = cl::Buffer(context.m_context, CL_MEM_READ_WRITE, size);
}


Buffer::Buffer(cl::Buffer buffer, std::shared_ptr<MemoryPool> allocator) :
	m_buffer(buffer),
	m_allocator(allocator)
{
	m_size = buffer.getInfo<CL_MEM_SIZE>();
}


Buffer::~Buffer()
{
	if (m_allocator != nullptr)
		m_allocator->hold(this);
}


std::shared_ptr<Buffer> Buffer::getSubRegion(size_t origin, size_t size)
{
	cl_buffer_region region = { origin, size };

	cl::Buffer sub = m_buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
	return std::make_shared<Buffer>(sub);
}


std::shared_ptr<Buffer> Buffer::getItem(py::slice slice)
{
	size_t start = slice.attr("start").cast<size_t>();
	size_t end = slice.attr("stop").cast<size_t>();

	return getSubRegion(start, end - start);
}


void enqueueCopy1D(CommandQueue* queue, Buffer* dest, Buffer* src, size_t byteCount,
	size_t destOffset, size_t srcOffset)
{
	queue->m_queue.enqueueCopyBuffer(src->m_buffer, dest->m_buffer, srcOffset, destOffset, byteCount);
}


void enqueueCopy3D(CommandQueue* queue, Buffer* dest, Buffer* src, py::tuple destOrigin, py::tuple srcOrigin,
	py::tuple region, py::tuple destPitches, py::tuple srcPitches)
{
	std::array<size_t, 3> vDestOrigin, vSrcOrigin, vRegion;
	size_t srcRowPitch, srcSlicePitch, destRowPitch, destSlicePitch;

	if (destOrigin.size() != 3)
		throw std::runtime_error("Invalid dest origin size");

	for (int i = 0; i < 3; i++)
		vDestOrigin[i] = destOrigin[i].cast<size_t>();

	if (srcOrigin.size() != 3)
		throw std::runtime_error("Invalid src origin size");

	for (int i = 0; i < 3; i++)
		vSrcOrigin[i] = srcOrigin[i].cast<size_t>();

	if (region.size() != 3)
		throw std::runtime_error("Invalid region size");

	for (int i = 0; i < 3; i++)
		vRegion[i] = region[i].cast<size_t>();

	if (destPitches.size() != 2)
		throw std::runtime_error("Invalid dest pitches size");

	destRowPitch = destPitches[0].cast<size_t>();
	destSlicePitch = destPitches[1].cast<size_t>();

	if (srcPitches.size() != 2)
		throw std::runtime_error("Invalid src pitches size");

	srcRowPitch = srcPitches[0].cast<size_t>();
	srcSlicePitch = srcPitches[1].cast<size_t>();

	queue->m_queue.enqueueCopyBufferRect(src->m_buffer, dest->m_buffer, vSrcOrigin, vDestOrigin, vRegion, srcRowPitch,
		srcSlicePitch, destRowPitch, destSlicePitch);
}


void enqueueFillBuffer8B(CommandQueue* queue, Buffer* dest, int byte, size_t size, size_t offset)
{
	queue->m_queue.enqueueFillBuffer(dest->m_buffer, static_cast<signed char>(byte), offset, size);
}


void initBuffer(py::module &m)
{
	py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
		.def(py::init<Context, cl::size_type>(), py::arg("context"), py::arg("size"))
		.def_property_readonly("size", &Buffer::size)
		.def_property_readonly("allocator", &Buffer::allocator)
		.def_property_readonly("int_ptr", &Buffer::intPtr)
		.def("get_sub_region", &Buffer::getSubRegion, py::arg("origin"), py::arg("size"))
		.def("__getitem__", &Buffer::getItem);

	m.def("enqueue_copy_1d", &enqueueCopy1D, py::arg("queue").none(false), py::arg("dest"), py::arg("src"),
		py::arg("byte_count"), py::arg("dest_offset"), py::arg("src_offset"));
	m.def("enqueue_copy_3d", &enqueueCopy3D, py::arg("queue").none(false), py::arg("dest"), py::arg("src"),
		py::arg("dest_origin"), py::arg("src_origin"), py::arg("region"),
		py::arg("dest_pitches"), py::arg("src_pitches"));
	m.def("enqueue_fill_buffer_8b", &enqueueFillBuffer8B, py::arg("queue").none(false), py::arg("dest"),
		py::arg("byte"), py::arg("size"), py::arg("offset"));
}
