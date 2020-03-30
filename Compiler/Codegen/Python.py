import os
from string import Template

from PuzzleLib.Compiler.Codegen.Types import void_t, float_t, double_t, ptrdiff_t, Py_ssize_t
from PuzzleLib.Compiler.Codegen.Types import schar_t, short_t, int_t, llong_t, int8_t, int16_t, int32_t, int64_t
from PuzzleLib.Compiler.Codegen.Types import uchar_t, ushort_t, uint_t, ullong_t, uint8_t, uint16_t, uint32_t, uint64_t

from PuzzleLib.Compiler.Toolchain import guessToolchain
from PuzzleLib.Compiler.JIT import extensionFromString, stdCachePath


funcTmpl = Template("""
static PyObject *${modname}_$funcname(PyObject *self, PyObject *args)
{
	(void)self;
	$parsercode
	$valinit

$gilstart
	$value$funcname($callparams);
$gilend

	$retval;
}
""")


parserTmpl = Template("""
	$header

	if (!PyArg_ParseTuple(args, "$parsestr", $parseparams))
		return NULL;

	$footer
""")


modTmpl = Template("""
$bindings

static PyMethodDef methods[] = {
	$methods,
	{NULL, NULL, 0, NULL}
};


static PyModuleDef mod = {
	PyModuleDef_HEAD_INIT,
	.m_name = "$modname",
	.m_methods = methods
};


PyMODINIT_FUNC PyInit_$modname(void)
{
	$modinit
	return PyModule_Create(&mod);
}
""")


c2py = {
	schar_t: "b",
	short_t: "h",
	int_t: "i",
	llong_t: "L",

	uchar_t: "B",
	ushort_t: "H",
	uint_t: "I",
	ullong_t: "K",

	float_t: "f",
	double_t: "d",
	Py_ssize_t: "n"
}


c2native = {
	int8_t: schar_t,
	int16_t: short_t,
	int32_t: int_t,
	int64_t: llong_t,

	uint8_t: uchar_t,
	uint16_t: ushort_t,
	uint32_t: uint_t,
	uint64_t: ullong_t,

	float_t: float_t,
	double_t: double_t,
	ptrdiff_t: Py_ssize_t
}


class ParamParser:
	def __init__(self, totalargs):
		self.header, self.footer = [], []
		self.parsestr, self.parseparams = [], []
		self.callparams = []

		self.argindex = 0
		self.totalargs = totalargs


def defaultConverter(T, name, parser):
	objname = "%sObj" % name
	nativetype = c2native[T.aliasBase]

	parser.header.append("%s;" % (nativetype.typegen(asDecl=True) % objname))

	parser.parsestr.append(c2py[nativetype])
	parser.parseparams.append("&%s" % objname)

	parser.footer.append("%s = %s;" % (T.typegen(asDecl=True) % name, objname))
	parser.callparams.append(name)


def generatePythonBinding(modname, functions, converter=defaultConverter, finalizer=None, modinit=""):
	bindings = (generateFunctionParamParser(modname, func, converter, finalizer) for func in functions)
	methods = (
		"{\"%s\", %s_%s, %s, NULL}" % (
			funcname, modname, funcname, "METH_VARARGS" if len(arguments) > 0 else "METH_NOARGS"
		) for funcname, _, arguments, *_ in functions
	)

	module = modTmpl.substitute(
		modname=modname, bindings="\n".join(bindings), methods=",\n\t".join(methods), modinit=modinit
	)
	return module


def generateFunctionParamParser(modname, func, converter=defaultConverter, finalizer=None):
	funcname, returntype, arguments, *args = func
	parser = ParamParser(totalargs=len(arguments))

	gilstart, gilend = ("Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS") if len(args) > 0 and args[0] else ("", "")

	if len(arguments) == 0:
		parsercode = "(void)args;\n"

	else:
		for i, (T, name) in enumerate(arguments):
			parser.argindex = i
			converter(T, name, parser)

		if finalizer is not None:
			finalizer(funcname, parser)

		parsercode = parserTmpl.substitute(
			header="\n\t".join(parser.header), footer="\n\t".join(parser.footer),
			parsestr="".join(parser.parsestr), parseparams=", ".join(parser.parseparams)
		)

	if returntype is void_t:
		valinit, value, retval = "", "", "Py_RETURN_NONE"
	else:
		nativetype = c2native[returntype.aliasBase]

		valinit, value = "%s retval;" % nativetype.typegen(asDecl=False), "retval = "
		retval = "return Py_BuildValue(\"%s\", retval)" % c2py[nativetype]

	func = funcTmpl.substitute(
		modname=modname, funcname=funcname, parsercode=parsercode, callparams=", ".join(parser.callparams),
		valinit=valinit, value=value, retval=retval, gilstart=gilstart, gilend=gilend
	)

	return func


def unittest():
	functions = """

static int32_t test(int32_t a, int32_t b)
{
	return a + b;
}


static void test2(void)
{

}

"""

	binding = generatePythonBinding("module", [
		("test", int32_t, [(int32_t, "a"), (int32_t, "b")], True),
		("test2", void_t, [])
	])

	source = "#include <Python.h>\n%s%s" % (functions, binding)
	mod = extensionFromString(
		guessToolchain(verbose=2), "module", source, cachepath=os.path.join(stdCachePath, "tests"), recompile=True
	)

	assert mod.test(2, 2) == 4
	assert mod.test2() is None


if __name__ == "__main__":
	unittest()
