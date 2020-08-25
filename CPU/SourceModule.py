import sys, os
from string import Template

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Compiler.Codegen.Types import PointerType, void_t, ptrdiff_t, float_t, double_t
from PuzzleLib.Compiler.Codegen.Types import int8_t, int16_t, int32_t, int64_t
from PuzzleLib.Compiler.Codegen.Types import uint8_t, uint16_t, uint32_t, uint64_t

from PuzzleLib.Compiler.Codegen.Python import generatePythonBinding, defaultConverter
from PuzzleLib.Compiler.Toolchain import guessToolchain
from PuzzleLib.Compiler.JIT import extensionFromString

from PuzzleLib.CPU.CPUArray import CPUArray


class SourceModule:
	cToNumpy = {
		int8_t: "NPY_INT8",
		int16_t: "NPY_INT16",
		int32_t: "NPY_INT32",
		int64_t: "NPY_INT64",

		uint8_t: "NPY_UINT8",
		uint16_t: "NPY_UINT16",
		uint32_t: "NPY_UINT32",
		uint64_t: "NPY_UINT64",

		float_t: "NPY_FLOAT",
		double_t: "NPY_DOUBLE"
	}

	layoutChecker = """if (PyArray_TYPE(%s) != %s || !PyArray_IS_C_CONTIGUOUS(%s))
	{
		PyErr_SetString(PyExc_ValueError, "tensor #%s has wrong data layout");
		return NULL;
	}"""


	def __init__(self, source, functions, converter=None, finalizer=None, debug=False):
		self.source, self.functions = source, functions
		self.mod, self.debug = None, debug

		self.aryIndex = 0

		self.converter = converter if converter is not None else self.paramConverter
		self.finalizer = finalizer


	def paramConverter(self, T, name, parser):
		if isinstance(T.aliasBase, PointerType):
			self.aryIndex += 1
			decl, objname = T.typegen(asDecl=True) % name, "%sObj" % name

			parser.header.append("PyArrayObject *%s;" % objname)

			parser.parsestr.append("O!")
			parser.parseparams.extend(["&PyArray_Type", "&%s" % objname])

			parser.footer.extend([
				self.layoutChecker % (objname, self.cToNumpy[T.aliasBase.basetype], objname, self.aryIndex),
				"%s = PyArray_DATA(%s);" % (decl, objname), ""
			])

			parser.callparams.append(name)

		else:
			defaultConverter(T, name, parser)

		if parser.argindex + 1 == parser.totalargs:
			self.aryIndex = 0


	def generateSource(self, modname):
		header = """
#include <stddef.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

"""

		modinit = "import_array();"

		binding = generatePythonBinding(
			modname, self.functions, converter=self.converter, finalizer=self.finalizer, modinit=modinit
		)

		source = "%s%s%s" % (header, self.source, binding)
		return source


	def build(self):
		if self.mod is not None:
			return

		modname = "module"
		source = self.generateSource(modname)

		toolchain = guessToolchain(verbose=1).withOptimizationLevel(
			level=0 if self.debug else 4,
			debuglevel=3 if self.debug else 0
		).addLibrary("numpy", [np.get_include()], [], [])

		if sys.platform != "win32":
			toolchain = toolchain.addLibrary("math", [], [], ["m"])

		cachepath = os.path.join(Config.libname, Config.Backend.cpu.name)
		self.mod = extensionFromString(toolchain, modname, source, cachepath=cachepath, cleanup=False)


	def getFunction(self, name):
		self.build()
		return getattr(self.mod, name)


	def __getattr__(self, name):
		return self.getFunction(name)


class Kernel:
	def __init__(self, debug=False):
		self.module, self.debug = None, debug


	def generateSource(self):
		raise NotImplementedError()


class ElementwiseKernel(Kernel):
	eltwiseTmpl = Template("""

static void $name($arguments, ptrdiff_t size)
{
	for (ptrdiff_t i = 0; i < size; i++)
	{
		$operation;
	}
}


static void ${name}_strided($arguments, ptrdiff_t start, ptrdiff_t stop, ptrdiff_t step)
{
	for (ptrdiff_t i = start; i < stop; i += step)
	{
		$operation;
	}
}

""")


	def __init__(self, arguments, operation, name, debug=False):
		super().__init__(debug)

		self.arguments, self.operation, self.name = arguments, operation, name
		self.foundArray = False


	def generateSource(self):
		arguments = [(T.restrict if isinstance(T, PointerType) else T, name) for T, name in self.arguments]

		source = self.eltwiseTmpl.substitute(
			arguments=", ".join(T.typegen(asDecl=True) % name for T, name in arguments),
			operation=self.operation, name=self.name
		)

		functions = [
			(self.name, void_t, arguments, True),
			("%s_strided" % self.name, void_t, arguments, True)
		]

		return source, functions


	def paramConverter(self, T, name, parser):
		if not self.foundArray and isinstance(T.aliasBase, PointerType):
			parser.footer.append("npy_intp size = PyArray_SIZE(%sObj);" % name)
			self.foundArray = True

		self.module.paramConverter(T, name, parser)

		if parser.argindex + 1 == parser.totalargs:
			self.foundArray = False


	@staticmethod
	def funcFinalizer(name, parser):
		if name.endswith("strided"):
			parser.header.append("Py_ssize_t start, stop, step;")

			parser.parsestr.append("(nnn)")
			parser.parseparams.extend(["&start", "&stop", "&step"])

			parser.callparams.extend(["start", "size > stop ? stop : size", "step"])

		else:
			parser.callparams.append("size")


	def __call__(self, *args, **kwargs):
		if self.module is None:
			source, functions = self.generateSource()
			self.module = SourceModule(
				source, functions, converter=self.paramConverter,finalizer=self.funcFinalizer, debug=self.debug
			)

		func = getattr(self.module, self.name)
		func(*(arg.data if isinstance(arg, CPUArray) else arg for arg in args))


class ReductionKernel(Kernel):
	reduceTmpl = Template("""

#define READ_AND_MAP(i) ($mapExpr)
#define REDUCE(a, b) ($reduceExpr)


static $outtype reduction($arguments, ptrdiff_t size)
{
	$outtype acc = $neutral;

	for (ptrdiff_t i = 0; i < size; i++)
		acc = REDUCE(acc, READ_AND_MAP(i));

	return acc;
}

""")


	np2c = {
		np.int8: int8_t,
		np.int16: int16_t,
		np.int32: int32_t,
		np.int64: int64_t,
		np.float32: float_t,
		np.float64: double_t
	}


	def __init__(self, outtype, neutral, reduceExpr, mapExpr, arguments, debug=False):
		super().__init__(debug)

		self.outtype, self.neutral = outtype, neutral
		self.reduceExpr, self.mapExpr, self.arguments = reduceExpr, mapExpr, arguments

		self.foundArray = False


	def generateSource(self):
		arguments = [(T.restrict if isinstance(T, PointerType) else T, name) for T, name in self.arguments]

		source = self.reduceTmpl.substitute(
			outtype = self.np2c[self.outtype].typegen(asDecl=False), neutral=self.neutral,
			arguments=", ".join(T.typegen(asDecl=True) % name for T, name in arguments),
			reduceExpr=self.reduceExpr, mapExpr=self.mapExpr
		)

		functions = [("reduction", self.np2c[self.outtype], arguments, True)]
		return source, functions


	def paramConverter(self, T, name, parser):
		if not self.foundArray and isinstance(T.aliasBase, PointerType):
			parser.footer.append("ptrdiff_t size = PyArray_SIZE(%sObj);" % name)
			self.foundArray = True

		self.module.paramConverter(T, name, parser)

		if parser.argindex + 1 == parser.totalargs:
			self.foundArray = False


	@staticmethod
	def funcFinalizer(_, parser):
		parser.callparams.append("size")


	def __call__(self, *args, **kwargs):
		if self.module is None:
			source, functions = self.generateSource()
			self.module = SourceModule(
				source, functions, converter=self.paramConverter, finalizer=self.funcFinalizer, debug=self.debug
			)

		acc = self.module.reduction(*(arg.data if isinstance(arg, CPUArray) else arg for arg in args))

		result = CPUArray.empty((), self.outtype)
		result.fill(acc)

		return result


def unittest():
	moduleTest()
	eltwiseTest()
	reductionTest()


def moduleTest():
	outdata = np.empty((10, ), dtype=np.float32)
	indata = np.random.randn(10).astype(np.float32)

	module = SourceModule("""

static void square(float * __restrict outdata, const float * __restrict indata, ptrdiff_t size)
{
	for (ptrdiff_t i = 0; i < size; i++)
		outdata[i] = indata[i] * indata[i];
}

""", functions=[
		("square", void_t, [
			(float_t.ptr.restrict, "outdata"), (float_t.const.ptr.restrict, "indata"), (ptrdiff_t, "size")
		], True)
	])

	module.square(outdata, indata, outdata.size)
	assert np.allclose(indata * indata, outdata)


def eltwiseTest():
	outdata = CPUArray.empty((10, ), dtype=np.float32)
	indata = CPUArray.toDevice(np.random.randn(10).astype(np.float32))

	square = ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = indata[i] * indata[i]",
		"square"
	)

	square(outdata, indata)

	hostInData = indata.get()
	hostOutData = hostInData * hostInData

	assert np.allclose(hostOutData, outdata.get())


def reductionTest():
	data = CPUArray.toDevice(np.random.randn(10).astype(np.float32))

	accumulate = ReductionKernel(
		np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="data[i]",
		arguments=[(float_t.const.ptr, "data")]
	)

	acc = accumulate(data)

	hostSum = np.sum(data.get())
	assert np.allclose(hostSum, acc.get())


if __name__ == "__main__":
	unittest()
