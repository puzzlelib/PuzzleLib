#include "Queue.h"


CommandQueue::CommandQueue(Context context, bool profiling) :
	m_profiling(profiling),
	m_context(context)
{
	cl::QueueProperties props = profiling ? cl::QueueProperties::Profiling : cl::QueueProperties::None;
	m_queue = cl::CommandQueue(context.m_context, static_cast<cl_command_queue_properties>(props), NULL);
}


void CommandQueue::flush()
{
	m_queue.flush();
}


void CommandQueue::finish()
{
	m_queue.finish();
}


void enqueueBarrier(CommandQueue* queue)
{
	queue->m_queue.enqueueBarrierWithWaitList();
}


void initCommandQueue(py::module &m)
{
	py::class_<CommandQueue>(m, "CommandQueue")
		.def(py::init<Context, bool>(), py::arg("context"), py::arg("profiling") = true)
		.def_property_readonly("int_ptr", &CommandQueue::intPtr)
		.def_property_readonly("profiling", &CommandQueue::profiling)
		.def("flush", &CommandQueue::flush)
		.def("finish", &CommandQueue::finish);

	m.def("enqueue_barrier", &enqueueBarrier, py::arg("queue"));
}
