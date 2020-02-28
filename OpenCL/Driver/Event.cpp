#include "Event.h"


Event::Event(cl::Event ev) :
	m_event(ev)
{

}


void Event::wait()
{
	m_event.wait();
}


py::tuple Event::profile()
{
	py::tuple tm(2);
	tm[0] = m_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	tm[1] = m_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

	return tm;
}


Event enqueueMarker(CommandQueue queue)
{
	cl::Event ev;
	queue.m_queue.enqueueMarkerWithWaitList(nullptr, &ev);
	return Event(ev);
}


void initEvent(py::module &m)
{
	py::class_<Event>(m, "Event")
		.def("wait", &Event::wait)
		.def("profile", &Event::profile);

	m.def("enqueue_marker", &enqueueMarker);
}
