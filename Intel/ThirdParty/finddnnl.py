import sys, os


def findDNNL():
	versions = ["1.91", "1.2", "1.1"]
	error = ""

	if sys.platform == "linux":
		libnames = ["libdnnl.so.%s" % v for v in versions]
		libnames += ["/usr/local/lib/%s" % libname for libname in libnames]

	elif sys.platform == "darwin":
		libnames = ["libdnnl.%s.dylib" % v for v in versions]

	elif sys.platform == "win32":
		libpaths = [
			os.environ.get("DNNL_PATH", ""),
			os.path.normpath(os.path.join(os.path.dirname(__file__), "../Libs/"))
		]

		libnames = [os.path.join(lp, "dnnl.dll") for lp in libpaths]
		error = ": check your DNNL_PATH environment value"

	else:
		raise RuntimeError("Unsupported platform for dnnl")

	for libname in libnames:
		if os.path.exists(libname):
			return libname

	raise OSError("dnnl library not found (searched for following version(s): %s)%s" % (versions, error))
