import time


def timeKernel(func, args, kwargs=None, looplength=1000, log=True, logname=None, normalize=False, hotpass=True):
	if kwargs is None:
		kwargs = {}

	if hotpass:
		func(*args, **kwargs)

	hostStart = time.time()

	for _ in range(looplength):
		func(*args, **kwargs)

	hostEnd = time.time()
	hostsecs = hostEnd - hostStart

	if logname is None:
		if hasattr(func, "__name__"):
			logname = "%s.%s" % (func.__module__, func.__name__)
		else:
			logname = "%s.%s" % (func.__module__ , func.__class__.__name__)

	if normalize:
		hostsecs /= looplength

	if log:
		print("%s host time: %s secs" % (logname, hostsecs))

	return hostsecs
