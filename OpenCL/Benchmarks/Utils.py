import time

from PuzzleLib.OpenCL.Driver import Driver
from PuzzleLib.OpenCL.Utils import queue


def timeKernel(func, args, kwargs=None, looplength=1000, log=True, logname=None, normalize=False, hotpass=True):
	if kwargs is None:
		kwargs = {}

	if hotpass:
		func(*args, **kwargs)

	hostStart = time.time()
	start = Driver.enqueue_marker(queue)

	for _ in range(looplength):
		func(*args, **kwargs)

	end = Driver.enqueue_marker(queue)
	hostEnd = time.time()

	end.wait()

	nanosInSec = 1e-9
	devsecs = (end.profile()[0] - start.profile()[1]) * nanosInSec
	hostsecs = hostEnd - hostStart

	if logname is None:
		if hasattr(func, "__name__"):
			logname = "%s.%s" % (func.__module__, func.__name__)
		else:
			logname = "%s.%s" % (func.__module__ , func.__class__.__name__)

	if normalize:
		devsecs /= looplength
		hostsecs /= looplength

	if log:
		print("%s device time: %s secs" % (logname, devsecs))
		print("%s host time: %s secs" % (logname, hostsecs))

	return devsecs, hostsecs
