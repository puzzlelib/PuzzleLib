import os, subprocess, itertools

from PuzzleLib.Compiler.Compilers.Compiler import Compiler, CompilerError


class MSVC(Compiler):
	cc = "cl"

	cflags = ["/W4"]
	ldflags = ["/LD"]

	vcenv = None


	def objectLine(self, extfile, sourcefiles):
		return self.fullCFlags(asObject=True) + self.outflags(extfile, asObject=True) + sourcefiles


	def linkLine(self, extfile, objfiles):
		return self.fullLDFlags() + self.outflags(extfile, asObject=False) + objfiles + self.linkFlags(extfile)


	def buildLine(self, extfile, sourcefiles):
		return self.fullCFlags(asObject=False) + self.fullLDFlags() + self.outflags(extfile, asObject=False) + \
			   ["/MP"] + sourcefiles + self.linkFlags(extfile)


	def depLine(self, sourcefiles):
		return ["/showIncludes"] + self.fullCFlags(asObject=True, debug=False, optimize=False) + \
			   self.outflags(None, asObject=True, debug=False, optimize=False) + sourcefiles


	def fullCFlags(self, asObject, debug=True, optimize=True):
		oflags = ["/std:c++14", "/EHsc"] if self.cpp else []

		if optimize and self.optlevel > 0:
			oflags.append("/Ox" if self.optlevel >= 3 else "/O%s" % self.optlevel)

			if self.optlevel >= 3:
				oflags.append("/fp:fast")

			if debug and self.debuglevel >= 3:
				oflags.append("/Oy-")

		oflags.extend("/D%s" % define for define in self.defines)
		return self.cflags + oflags + ["/I%s" % idir for idir in self.includeDirs] + (["/c"] if asObject else [])


	def fullLDFlags(self):
		return self.ldflags + ["%s%s" % (lib, self.libext) for lib in self.libraries]


	def outflags(self, extfile, asObject, debug=True, optimize=True):
		if extfile is None:
			return ["/nologo", "/FoNUL"]

		elif asObject:
			outpath, dbgpath = extfile, os.path.dirname(extfile) + os.path.sep
		else:
			outpath = dbgpath = os.path.dirname(extfile) + os.path.sep

		outflags = ["/nologo", "/Fo%s" % outpath]

		if debug:
			if self.debuglevel > 0:
				outflags.extend(["/Fd%s" % dbgpath, "/ZI" if self.debuglevel >= 3 else "/Zi"])

		if optimize and self.debuglevel == 0 and self.optlevel >= 4:
			outflags.append("/GL")

		return outflags


	def linkFlags(self, extfile):
		return ["/link", "/IMPLIB:%s" % (os.path.splitext(extfile)[0] + self.libext), "/OUT:%s" % extfile] + \
			   ["/LIBPATH:%s" % ldir for ldir in self.libraryDirs]


	def invoke(self, cmdline, asLinker=False, verbose=None):
		verbose = self.verbose if verbose is None else verbose

		if self.env is None:
			self.env = self.createEnvironment(verbose)

		return super().invoke(cmdline, asLinker, verbose)


	@classmethod
	def createEnvironment(cls, verbose):
		if cls.vcenv is None:
			if verbose > 1:
				print("Creating msvc environment ...", flush=True)

			cls.vcenv = getEnv("amd64")

		return cls.vcenv


def getEnv(platspec):
	vcvarsall = findVCVarsAll()

	if vcvarsall is None:
		raise CompilerError("Unable to find vcvarsall.bat")

	try:
		out = subprocess.check_output(
			"cmd /u /c \"%s\" %s && set" % (vcvarsall, platspec), stderr=subprocess.STDOUT
		).decode("utf-16le", errors="replace")

	except subprocess.CalledProcessError as exc:
		raise CompilerError("Error executing %s" % exc.cmd)

	return {
		key.lower(): value for key, _, value in (line.partition("=") for line in out.splitlines()) if key and value
	}


def findVCVarsAll():
	vcpath = findVC2017()
	vcpath = findVC2015() if vcpath is None else vcpath

	if vcpath is None:
		return None

	vcvarsall = os.path.join(vcpath, "vcvarsall.bat")
	return vcvarsall if os.path.isfile(vcvarsall) else None


def findVC2017():
	root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")

	if root is None:
		return None

	try:
		vcpath = subprocess.check_output([
			os.path.join(root, "Microsoft Visual Studio", "Installer", "vswhere.exe"),
			"-latest", "-prerelease",
			"-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
			"-property", "installationPath"
		], encoding="mbcs", errors="strict").strip()

	except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
		return None

	vcpath = os.path.join(vcpath, "VC", "Auxiliary", "Build")
	return vcpath if os.path.isdir(vcpath) else None


def findVC2015():
	import winreg

	try:
		key = winreg.OpenKeyEx(
			winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\VisualStudio\SxS\VC7",
			access=winreg.KEY_READ | winreg.KEY_WOW64_32KEY
		)

	except OSError:
		return None

	with key:
		vcversion, vcpath = 0, None

		for i in itertools.count():
			try:
				v, path, vt = winreg.EnumValue(key, i)

			except OSError:
				break

			if v and vt == winreg.REG_SZ and os.path.isdir(path):
				try:
					version = int(v)

				except (ValueError, TypeError):
					continue

				if version >= 14 and version > vcversion:
					vcversion, vcpath = version, path

	return vcpath
