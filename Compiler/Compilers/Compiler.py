import sys, os, subprocess, sysconfig


class CompilerError(Exception):
	pass


class Compiler:
	cc = None

	cflags = None
	ldflags = None


	def __init__(self, verbose=0, env=None, forPython=True):
		self.verbose, self.env = verbose, env

		platform = self.setupPlatform(forPython)

		self.pydext, self.oext, self.libext, self.soext, self.linkext, self.debugext = platform[:6]
		self.includeDirs, self.libraryDirs, self.libraries = platform[6:]

		self.features = set()
		self.defines = set()

		self.optlevel = 0
		self.debuglevel = 0

		self.cpp = False

		self.keys = (
			"cc", "cflags", "ldflags", "features", "includeDirs", "libraryDirs", "libraries", "defines",
			"optlevel", "debuglevel", "cpp"
		)


	@staticmethod
	def setupPlatform(forPython):
		config = sysconfig.get_config_vars()
		pydext = config["EXT_SUFFIX"]

		includeDirs = [config["INCLUDEPY"]] if forPython else []
		libraryDirs, libraries = [], []

		if sys.platform == "win32":
			oext, libext, soext = ".obj", ".lib", ".dll"
			linkext, debugext = [".exp"], [".pdb", ".idb", ".ilk"]

			if forPython:
				bindir = config["BINDIR"]

				libraryDirs = [os.path.join(bindir, "libs"), bindir]
				libraries = ["python%s%s" % sys.version_info[:2]]

		elif sys.platform in ("linux", "darwin"):
			oext, libext, soext = ".o", ".a", (".so" if sys.platform == "linux" else ".dylib")
			linkext, debugext = [], []

			if forPython:
				libraryDirs = [config["LIBDIR"]]

				(major, minor), mext = sys.version_info[:2], "m" if sys.platform == "linux" else ""
				libraries = ["python%s.%s%s" % (major, minor, mext if minor < 8 else "")]

		else:
			raise NotImplementedError(sys.platform)

		return pydext, oext, libext, soext, linkext, debugext, includeDirs, libraryDirs, libraries


	def cppMode(self, enabled):
		self.cpp = enabled
		return self


	def addLibrary(self, name, includeDirs, libraryDirs, libraries):
		if name in self.features:
			return

		self.features.add(name)

		self.includeDirs.extend(os.path.normpath(path) for path in includeDirs)
		self.libraryDirs.extend(os.path.normpath(path) for path in libraryDirs)

		self.libraries.extend(libraries)
		return self


	def addDefine(self, *defines):
		self.defines.update(defines)
		return self


	def clearPath(self, path):
		exts = [self.oext, self.libext] + self.linkext

		if self.debuglevel == 0:
			exts.extend(self.debugext)

		for file in filter(lambda f: any(f.endswith(ext) for ext in exts), os.listdir(path)):
			os.remove(os.path.join(path, file))

		return self


	def objectLine(self, extfile, sourcefiles):
		raise NotImplementedError()


	def linkLine(self, extfile, objfiles):
		raise NotImplementedError()


	def buildLine(self, extfile, sourcefiles):
		raise NotImplementedError()


	def depLine(self, sourcefiles):
		raise NotImplementedError()


	def buildObject(self, extfile, sourcefiles):
		sourcefiles = [sourcefiles] if not isinstance(sourcefiles, list) else sourcefiles
		self.invoke(self.cmdline(self.objectLine(extfile, sourcefiles)))

		return self


	def link(self, extfile, objfiles):
		objfiles = [objfiles] if not isinstance(objfiles, list) else objfiles
		self.invoke(self.cmdline(self.linkLine(extfile, objfiles)), asLinker=True)

		return self


	def build(self, extfile, sourcefiles):
		sourcefiles = [sourcefiles] if not isinstance(sourcefiles, list) else sourcefiles
		self.invoke(self.cmdline(self.buildLine(extfile, sourcefiles)))

		return self


	def getDependencies(self, sourcefiles, cwd):
		sourcefiles = [sourcefiles] if not isinstance(sourcefiles, list) else sourcefiles
		deps = self.invoke(self.cmdline(self.depLine(sourcefiles)), verbose=self.verbose - 1).decode()

		files = []
		for line in deps.splitlines():
			files.extend(self.extractPathsFromLine(line))

		cwd = os.path.normcase(os.path.realpath(cwd))

		deps = set(file for file in files if os.path.commonprefix([file, cwd]) == cwd)
		deps.update(os.path.normcase(os.path.realpath(file)) for file in sourcefiles)

		return deps


	def withOptimizationLevel(self, level=0, debuglevel=0):
		self.optlevel, self.debuglevel = level, debuglevel
		return self


	def invoke(self, cmdline, asLinker=False, verbose=None):
		verbose = self.verbose if verbose is None else verbose

		if verbose > 1:
			print("%s invocation: %s" % (self.getInvoked(asLinker), " ".join(cmdline)), flush=True)

		result = subprocess.run(cmdline, env=self.env, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

		if result.returncode != 0 or verbose > 0:
			streams = [
				stream.decode().strip() if stream is not None else "" for stream in [result.stderr, result.stdout]
			]

			text = "\n\n".join(stream for stream in streams if len(stream) > 0)

			if result.returncode != 0:
				raise CompilerError(
					"%s invocation failed: %s\n\n%s\n" % (self.getInvoked(asLinker), " ".join(cmdline), text)
				)

			if len(text) > 0 or verbose > 1:
				print("\n%s\n" % text if len(text) > 0 else "", flush=True)

		return result.stdout


	def cmdline(self, flags):
		return [self.cc] + flags


	@staticmethod
	def getInvoked(asLinker):
		return "Linker" if asLinker else "Compiler"


	@staticmethod
	def extractPathsFromLine(line):
		line = line.split(sep=": ")[-1].strip() if line[0] != " " else line

		paths = (path.strip() for path in line.split())
		return (os.path.normcase(os.path.abspath(path)) for path in paths if os.path.exists(path))


	@staticmethod
	def normalizeParam(value):
		if isinstance(value, set):
			return ",".join(value for value in sorted(value))

		elif isinstance(value, (list, tuple)):
			return ",".join(value)

		return value


	def signature(self):
		kv = (
			(key, self.normalizeParam(value)) for key, value in ((key, getattr(self, key)) for key in sorted(self.keys))
		)
		return ";".join("%s:%s" % (key, value) for key, value in kv)
