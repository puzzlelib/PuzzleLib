#ifndef QUEUE_H
#define QUEUE_H

#include "Common.h"
#include "Context.h"


struct CommandQueue
{
	cl::CommandQueue m_queue;
	bool m_profiling;
	Context m_context;


	CommandQueue(Context context, bool profiling=true);
	size_t intPtr() { return reinterpret_cast<size_t>(m_queue.get()); }
	bool profiling() { return m_profiling; }
	void flush();
	void finish();
};


void enqueueBarrier(CommandQueue* queue);


void initCommandQueue(py::module &m);


#endif
