import time

from PuzzleLib.Cuda.Driver import Event


def timeKernel(func, args, kwargs=None, looplength=1000, log=True, logname=None, normalize=False, hotpass=True):
	if kwargs is None:
		kwargs = {}

	if hotpass:
		func(*args, **kwargs)

	start, end = Event(), Event()

	hostStart = time.time()
	start.record()

	for _ in range(looplength):
		func(*args, **kwargs)

	end.record()
	hostEnd = time.time()

	end.synchronize()
	millisInSec = 1e-3

	devsecs = start.timeTill(end) * millisInSec
	hostsecs = hostEnd - hostStart

	if logname is None:
		funcname = func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
		logname = "%s.%s" % (func.__module__, funcname)

	if normalize:
		devsecs /= looplength
		hostsecs /= looplength

	if log:
		print("%s device time: %s secs" % (logname, devsecs))
		print("%s host time: %s secs" % (logname, hostsecs))

	return devsecs, hostsecs
