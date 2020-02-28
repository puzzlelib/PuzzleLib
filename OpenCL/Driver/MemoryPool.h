#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cstdint>
#include <map>
#include <vector>
#include <memory>

#include "Context.h"


struct Buffer;


struct MemoryPool : public std::enable_shared_from_this<MemoryPool>
{
	std::map<uint32_t, std::vector<cl::Buffer>> m_pool;
	Context m_context;
	bool m_holding;
	size_t m_activeBlocks;
	size_t m_heldBlocks;


	MemoryPool(Context context);
	void hold(Buffer* buffer);
	bool isHolding() { return m_holding; }
	size_t activeBlocks() { return m_activeBlocks; }
	size_t heldBlocks() { return m_heldBlocks; }
	void stopHolding();
	void freeHeld();
	uint32_t binNumber(size_t size);
	size_t allocSize(uint32_t bin);
	bool tryFreeMemory();
};


std::shared_ptr<Buffer> allocate(std::shared_ptr<MemoryPool> allocator, size_t size);


void initMemoryPool(py::module &m);


#endif
