import sys

from PuzzleLib.Compiler.Compilers.Compiler import Compiler


class GCCLike(Compiler):
	cflags = ["-Wall", "-Wextra"]
	ldflags = ["--shared"]


	def __init__(self, verbose, forPython=True):
		super().__init__(verbose, forPython=forPython)

		if sys.platform == "linux":
			self.cflags = self.cflags + ["-fPIC"]


	def objectLine(self, extfile, sourcefiles):
		return self.fullCFlags(asObject=True) + self.outFlags(extfile) + sourcefiles


	def linkLine(self, extfile, objfiles):
		return self.fullLDFlags() + self.outFlags(extfile) + objfiles + self.linkFlags()


	def buildLine(self, extfile, sourcefiles):
		return self.fullCFlags(asObject=False) + self.fullLDFlags() + self.outFlags(extfile) + \
			   sourcefiles + self.linkFlags()


	def depLine(self, sourcefiles):
		return ["-M"] + self.fullCFlags(asObject=False, debug=False, optimize=False) + sourcefiles


	def fullCFlags(self, asObject, debug=True, optimize=True):
		oflags = self.fullCppFlags()

		if debug and self.debuglevel > 0:
			oflags.append("-g3" if self.debuglevel >= 3 else "-g")

		if optimize and self.optlevel > 0:
			oflags.append("-O3" if self.optlevel >= 3 else "-O%s" % self.optlevel)

			if self.optlevel >= 3:
				oflags.extend(["-march=native", "-mtune=native", "-ffast-math"])

			if debug and self.debuglevel >= 3:
				oflags.append("-fno-omit-frame-pointer")

		oflags.extend("-D%s" % define for define in self.defines)
		return self.cflags + oflags + ["-I%s" % idir for idir in self.includeDirs] + (["-c"] if asObject else [])


	def fullCppFlags(self):
		return ["-std=c++14" if self.cpp else "-std=gnu99"]


	def fullLDFlags(self):
		return self.ldflags + ["-L%s" % ldir for ldir in self.libraryDirs]


	def outFlags(self, extfile):
		outFlags = ["-o", extfile]

		if self.optlevel >= 4:
			outFlags.append("-flto")

		return outFlags


	def linkFlags(self):
		return ["-l%s" % lib for lib in self.libraries]


class GCC(GCCLike):
	cc = "gcc"


class Clang(GCCLike):
	cc = "clang"


	def fullCppFlags(self):
		return ["-std=c++14" if self.cpp else "-std=c99"]


	def outFlags(self, extfile):
		outflags = super().outFlags(extfile)

		if sys.platform == "win32":
			outflags.append("-fuse-ld=lld")

		return outflags
