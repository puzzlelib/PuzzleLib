import sys

from PuzzleLib.Compiler.Compilers.GCC import GCCLike
from PuzzleLib.Compiler.Compilers.MSVC import MSVC


class NVCC(GCCLike):
	cc = "nvcc"


	def __init__(self, verbose=0, forPython=False):
		super().__init__(verbose)
		cflags = MSVC.cflags if sys.platform == "win32" else self.cflags

		self.cflags = [flag for cflag in cflags for flag in ["-Xcompiler", cflag]]
		self.cpp = True

		if not forPython:
			self.ldflags = []


	def cppMode(self, enabled):
		assert False


	def	fullCFlags(self, asObject, debug=True, optimize=True):
		oflags = self.fullCppFlags()

		if debug and self.debuglevel > 0:
			oflags.extend(["-g", "-G" if self.debuglevel >= 3 else "-lineinfo"])

		if optimize and self.optlevel > 0:
			oflags.append("-O3" if self.optlevel >= 3 else "-O%s" % self.optlevel)

			if self.optlevel >= 3:
				oflags.append("-use_fast_math")

		return self.cflags + oflags + ["-I%s" % idir for idir in self.includeDirs] + (["-c"] if asObject else [])


	def fullCppFlags(self):
		return [] if sys.platform == "win32" else ["-std=c++14"]


	def outFlags(self, extfile):
		return ["-o", extfile]
