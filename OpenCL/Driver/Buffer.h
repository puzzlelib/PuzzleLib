#ifndef BUFFER_H
#define BUFFER_H

#include "Common.h"
#include "Context.h"


struct MemoryPool;
struct CommandQueue;


struct Buffer : public std::enable_shared_from_this<Buffer>
{
	cl::Buffer m_buffer;
	std::shared_ptr<MemoryPool> m_allocator;
	cl::size_type m_size;


	Buffer(Context context, size_t size);
	Buffer(Context context, size_t size, std::shared_ptr<MemoryPool> allocator);
	Buffer(cl::Buffer buffer, std::shared_ptr<MemoryPool> allocator=std::shared_ptr<MemoryPool>());
	Buffer(const Buffer&) = delete;
	~Buffer();
	cl::size_type size() { return m_size; }
	std::shared_ptr<MemoryPool> allocator() { return m_allocator; }
	std::shared_ptr<Buffer> getSubRegion(size_t origin, size_t size);
	std::shared_ptr<Buffer> getItem(py::slice slice);
	size_t intPtr() { return reinterpret_cast<size_t>(m_buffer.get()); }
};


void enqueueCopy1D(CommandQueue* queue, Buffer* dest, Buffer* src, size_t byteCount,
	size_t destOffset, size_t srcOffset);
void enqueueCopy3D(CommandQueue* queue, Buffer* dest, Buffer* src, py::tuple destOrigin, py::tuple srcOrigin,
	py::tuple region, py::tuple destPitches, py::tuple srcPitches);
void enqueueFillBuffer8B(CommandQueue* queue, Buffer* dest, int byte, size_t size, size_t offset);


void initBuffer(py::module &m);


#endif
