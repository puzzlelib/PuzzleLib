import sys, os, stat, shutil, importlib.util

from PuzzleLib.Compiler.Compilers.GCC import GCC, Clang
from PuzzleLib.Compiler.Compilers.MSVC import MSVC
from PuzzleLib.Compiler.Compilers.NVCC import NVCC


def guessToolchain(verbose=0):
	if sys.platform == "win32":
		return MSVC(verbose)

	elif sys.platform == "linux":
		return GCC(verbose)

	elif sys.platform == "darwin":
		return Clang(verbose)

	else:
		raise NotImplementedError(sys.platform)


def guessNVCCToolchain(verbose=0, forPython=False):
	return NVCC(verbose, forPython)


def loadDynamicModule(extfile, modulename=None):
	modulename = os.path.basename(extfile).split(sep=".")[0] if modulename is None else modulename
	spec = importlib.util.spec_from_file_location(modulename, os.path.abspath(extfile))

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	return module


def createTemplateNames(name):
	return ["%s.gen%s" % (name, ext) for name, ext in [(name, ".h"), (name, ".c")]]


def writeTemplates(sources):
	for source, filename in sources:
		writeTemplate(source, filename)


def writeTemplate(source, filename):
	if os.path.exists(filename):
		os.chmod(filename, stat.S_IWUSR | stat.S_IREAD)

	with open(filename, mode="w", encoding="utf-8") as f:
		f.write(source)

	os.chmod(filename, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def buildTemplateTest(name, path, bindingName, generator, verbose=2, debuglevel=0, level=4, defines=None, **kwargs):
	sources = generator(name=name, filename=os.path.join(path, name), **kwargs)
	cc = guessToolchain(verbose=verbose).withOptimizationLevel(debuglevel=debuglevel, level=level)

	defines = [] if defines is None else defines
	cc.addDefine(*defines)

	binding = os.path.join(path, bindingName)
	if os.path.exists(binding):
		os.chmod(binding, stat.S_IWUSR | stat.S_IREAD)

	shutil.copy(bindingName, binding)
	os.chmod(binding, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

	sources = sources + [binding] if isinstance(sources, list) else [sources, binding]
	modname = os.path.join(path, name + cc.pydext)

	cc.build(modname, sources).clearPath(path)
	return loadDynamicModule(modname)
