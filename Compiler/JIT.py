import sys, os, hashlib, tempfile

from PuzzleLib.Compiler.Toolchain import guessToolchain, loadDynamicModule


class JITError(Exception):
	pass


def extensionFromString(toolchain, name, string, cachepath, cleanup=False, recompile=False, srcext=".c"):
	modulename, extfile = compileFromString(toolchain, name, string, cachepath, cleanup, recompile, srcext)
	return loadDynamicModule(extfile, modulename)


def compileFromString(toolchain, name, string, cachepath, cleanup, recompile, srcext):
	cachedir = getCacheDir(cachepath)
	sourcename = "%s%s" % (name, srcext)

	cmdline = toolchain.cmdline(toolchain.buildLine(name, [sourcename]))
	hashsum = computeHash(string, *cmdline)

	modulepath = os.path.join(cachedir, hashsum)
	modulename = "%s.%s" % (hashsum, name)

	extfile = os.path.join(modulepath, name + toolchain.pydext)
	sourcename = os.path.join(modulepath, sourcename)

	with FileLock(cachedir):
		if not os.path.exists(extfile) or recompile:
			if toolchain.verbose > 1:
				msg = (
					"### Forcing extension '%s' recompilation ..." if recompile else
					"### No cache found for extension '%s', performing compilation ..."
				) % name
				print(msg, flush=True)

			os.makedirs(modulepath, exist_ok=True)

			if cleanup:
				f = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=srcext, delete=False)
				try:
					with f:
						f.write(string)

					toolchain.build(extfile, f.name)

				finally:
					os.remove(f.name)
					toolchain.clearPath(modulepath)

			else:
				with open(sourcename, mode="w", encoding="utf-8") as f:
					f.write(string)

				toolchain.build(extfile, sourcename)

		elif toolchain.verbose > 1:
			print("### Found cached compilation for extension '%s', skipping compilation ..." % name, flush=True)

	return modulename, extfile


def getCacheDir(dirname):
	if sys.platform == "win32":
		path = os.path.normpath(os.environ["LOCALAPPDATA"])

	elif sys.platform == "linux":
		path = os.path.expanduser("~/.cache")

	elif sys.platform == "darwin":
		path = os.path.expanduser("~/Library/Caches")

	else:
		raise NotImplementedError(sys.platform)

	cachedir = os.path.join(path, dirname)
	os.makedirs(cachedir, exist_ok=True)

	return cachedir


def computeHash(*lines):
	hasher = hashlib.sha256()

	for line in lines:
		hasher.update(line.encode())

	return hasher.hexdigest()


class FileLock:
	def __init__(self, dirpath):
		self.lockfile = os.path.join(dirpath, "lock")

		self.dirpath = dirpath
		self.fd = None


	def __enter__(self):
		try:
			fd = os.open(self.lockfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC)

		except (IOError, OSError):
			raise JITError("Could not lock directory '%s'" % self.dirpath)

		self.fd = fd


	def __exit__(self, exc_type, exc_val, exc_tb):
		os.close(self.fd)
		self.fd = None

		try:
			os.remove(self.lockfile)

		except OSError:
			pass


def unittest():
	toolchain = guessToolchain(verbose=2).withOptimizationLevel(level=4)

	src = """
#include <Python.h>


static PyObject *hello(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	puts("Hello, JIT!");
	fflush(stdout);

	Py_RETURN_NONE;
}


static PyMethodDef methods[] = {
	{"hello", hello, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};


static PyModuleDef mod = {
	PyModuleDef_HEAD_INIT,
	.m_name = "test",
	.m_methods = methods
};


PyMODINIT_FUNC PyInit_test(void)
{
	return PyModule_Create(&mod);
}
"""

	test = extensionFromString(
		toolchain, name="test", string=src, cachepath=os.path.join("PuzzleLib", "tests"), cleanup=True, recompile=True
	)
	test.hello()


if __name__ == "__main__":
	unittest()
