import os
from PuzzleLib.Compiler.Toolchain import copySource


def generatePyDefines(path):
	dirname = os.path.dirname(__file__)
	copySource(os.path.join(dirname, "PyDefines.h"), os.path.join(path, "PyDefines.gen.h"))
