import sys, os, ctypes


def findDNNL():
	versions = ["1.91", "1.2", "1.1"]

	if sys.platform == "linux":
		libnames = ["libdnnl.so.%s" % v for v in versions]
		libnames += ["/usr/local/lib/%s" % libname for libname in libnames]

	elif sys.platform == "darwin":
		libnames = ["/usr/local/lib/libdnnl.%s.dylib" % v for v in versions]

	elif sys.platform == "win32":
		libpaths = [
			os.environ.get("DNNL_PATH", ""),
			os.path.normpath(os.path.join(os.path.dirname(__file__), "../Libs/"))
		]

		libnames = [os.path.join(libpath, "dnnl.dll") for libpath in libpaths]

	else:
		raise RuntimeError("Unsupported platform for dnnl")

	cloader = ctypes.windll if sys.platform == "win32" else ctypes.cdll

	for libname in libnames:
		try:
			clib = cloader.LoadLibrary(libname)

		except OSError:
			pass

		else:
			return libname, clib

	raise OSError("dnnl library not found (searched for following version(s): %s)" % versions)
