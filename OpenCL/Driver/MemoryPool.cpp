#include "MemoryPool.h"
#include "Buffer.h"


const uint32_t logTable8[] =
{
	0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};


inline uint32_t bitlog2_16(uint16_t v)
{
	if (unsigned long t = v >> 8)
		return 8 + logTable8[t];
	else
		return logTable8[v];
}


inline uint32_t bitlog2_32(uint32_t v)
{
	if (uint16_t t = v >> 16)
		return 16 + bitlog2_16(t);
	else
		return bitlog2_16(uint16_t(v));
}


inline uint32_t bitlog2(size_t v)
{
#if (ULONG_MAX != 4294967295) || defined(_WIN64)
	if (uint32_t t = v >> 32)
		return 32 + bitlog2_32(t);
	else
#endif
		return bitlog2_32(unsigned(v));
}


template <class T>
inline T signedLeftShift(T x, signed shift)
{
	if (shift < 0)
		return x >> -shift;
	else
		return x << shift;
}


template <class T>
inline T signedRightShift(T x, signed shift)
{
	if (shift < 0)
		return x << -shift;
	else
		return x >> shift;
}


const uint32_t mantissaBits = 2;
const uint32_t mantissaMask = (1 << mantissaBits) - 1;


MemoryPool::MemoryPool(Context context) :
	m_context(context),
	m_holding(true),
	m_activeBlocks(0),
	m_heldBlocks(0)
{}


std::shared_ptr<Buffer> allocate(std::shared_ptr<MemoryPool> allocator, size_t size)
{
	uint32_t bin = allocator->binNumber(size);
	auto p = allocator->m_pool.find(bin);

	if (p == allocator->m_pool.end() || p->second.size() == 0)
	{
		size_t allocSize = allocator->allocSize(bin);
		assert(allocator->binNumber(allocSize) == bin);

		do
		{
			try
			{
				auto buffer = std::make_shared<Buffer>(allocator->m_context, allocSize, allocator);
				allocator->m_activeBlocks++;

				return buffer;
			}
			catch (cl::Error&)
			{
				if (!allocator->tryFreeMemory())
					throw std::runtime_error("Out of memory");
			}
		}
		while (true);
	}

	auto buffer = p->second.back();
	p->second.pop_back();

	allocator->m_activeBlocks++;
	allocator->m_heldBlocks--;

	return std::make_shared<Buffer>(buffer, allocator);
}


void MemoryPool::hold(Buffer* buffer)
{
	m_activeBlocks--;

	if (!m_holding)
		return;

	uint32_t bin = binNumber(buffer->m_size);
	m_pool[bin].push_back(buffer->m_buffer);
	
	m_heldBlocks++;
}


void MemoryPool::stopHolding()
{
	m_holding = false;
}


void MemoryPool::freeHeld()
{
	m_pool.clear();
	m_heldBlocks = 0;
}


uint32_t MemoryPool::binNumber(size_t size)
{
	int32_t msb = bitlog2(size);
	int32_t shifted = static_cast<int32_t>(signedRightShift(size, msb - signed(mantissaBits)));
	assert(size == 0 || (shifted & (1 << mantissaBits)) != 0);

	int32_t chopped = shifted & mantissaMask;
	return msb << mantissaBits | chopped;
}


size_t MemoryPool::allocSize(uint32_t bin)
{
	uint32_t exponent = bin >> mantissaBits;
	uint32_t mantissa = bin & mantissaMask;

	size_t ones = signedLeftShift<size_t>(1, signed(exponent) - signed(mantissaBits));
	if (ones) ones -= 1;

	size_t head = signedLeftShift<size_t>((1 << mantissaBits) | mantissa, signed(exponent) - signed(mantissaBits));
	assert(!(ones & head));

	return head | ones;
}


bool MemoryPool::tryFreeMemory()
{
	for (auto it = m_pool.rend(); it != m_pool.rbegin(); it++)
	{
		if (it->second.size() > 0)
		{
			it->second.pop_back();
			m_heldBlocks--;
		}
	}

	return false;
}


void initMemoryPool(py::module &m)
{
	py::class_<MemoryPool, std::shared_ptr<MemoryPool>>(m, "MemoryPool")
		.def(py::init<Context>(), py::arg("context").none(false))
		.def_property_readonly("isHolding", &MemoryPool::isHolding)
		.def_property_readonly("activeBlocks", &MemoryPool::activeBlocks)
		.def_property_readonly("heldBlocks", &MemoryPool::heldBlocks)
		.def("allocate", &allocate, py::arg("size"))
		.def("stopHolding", &MemoryPool::stopHolding)
		.def("freeHeld", &MemoryPool::freeHeld);
}
