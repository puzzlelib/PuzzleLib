#include "Common.h"
#include "Event.h"
#include "Platform.h"
#include "Device.h"
#include "Context.h"
#include "Queue.h"
#include "Buffer.h"
#include "Array.h"
#include "MemoryPool.h"
#include "Program.h"


PYBIND11_MODULE(Driver, m)
{
	initCommon(m);
	initEvent(m);
	initPlatform(m);
	initDevice(m);
	initContext(m);
	initCommandQueue(m);
	initBuffer(m);
	initArray(m);
	initMemoryPool(m);
	initProgram(m);
}
