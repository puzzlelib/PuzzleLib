import sys, os, stat, shutil, importlib.util
from enum import Enum

from PuzzleLib.Compiler.Compilers.GCC import GCC, Clang
from PuzzleLib.Compiler.Compilers.MSVC import MSVC
from PuzzleLib.Compiler.Compilers.NVCC import NVCC


class Toolchain(Enum):
	msvc = "msvc"
	gcc = "gcc"
	clang = "clang"


def guessToolchain(verbose=0, tc=None):
	if tc is None:
		tc = {
			"win32": Toolchain.msvc,
			"linux": Toolchain.gcc,
			"darwin": Toolchain.clang
		}[sys.platform]

	cls = {
		Toolchain.msvc: MSVC,
		Toolchain.gcc: GCC,
		Toolchain.clang: Clang
	}[Toolchain(tc)]

	return cls(verbose=verbose)


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
		if os.path.exists(filename):
			os.chmod(filename, stat.S_IWUSR | stat.S_IREAD)

		with open(filename, mode="w", encoding="utf-8") as f:
			f.write(source)

		os.chmod(filename, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def copySource(src, dst):
	if os.path.exists(dst):
		os.chmod(dst, stat.S_IWUSR | stat.S_IREAD)

	shutil.copy(src, dst)
	os.chmod(dst, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def buildTemplateTest(name, path, bindingName, generator, verbose=2, debuglevel=0, level=4, defines=None, libs=None,
					  **kwargs):
	sources = generator(name=name, filename=os.path.join(path, name), **kwargs)

	binding = os.path.join(path, bindingName)
	copySource(bindingName, binding)

	sources = [sources, binding] if isinstance(sources, str) else list(sources) + [binding]
	return compileTemplateTest(os.path.join(path, name), sources, verbose, debuglevel, level, defines, libs)


def compileTemplateTest(modname, sources, verbose=2, debuglevel=0, level=4, defines=None, libs=None):
	cc = guessToolchain(verbose=verbose).withOptimizationLevel(debuglevel=debuglevel, level=level)

	defines = [] if defines is None else defines
	cc.addDefine(*defines)

	libs = [] if libs is None else libs
	for name, includeDirs, libraryDirs, libraries in libs:
		cc.addLibrary(name, includeDirs, libraryDirs, libraries)

	modname = modname + cc.pydext

	cc.build(modname, sources).clearPath(os.path.dirname(modname))
	return loadDynamicModule(modname)
