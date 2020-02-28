#ifndef EVENT_H
#define EVENT_H

#include "Common.h"
#include "Queue.h"


struct Event
{
	cl::Event m_event;


	Event(cl::Event ev);
	void wait();
	py::tuple profile();
};


Event enqueueMarker(CommandQueue queue);


void initEvent(py::module &m);


#endif
