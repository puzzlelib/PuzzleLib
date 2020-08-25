import os, stat, shutil, subprocess
from enum import Enum

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.sdist import sdist


libname = "PuzzleLib"
version = "1.0.2"


class Options(str, Enum):
	cuda = "cuda"
	hip = "hip"
	intel = "intel"
	tensorrt = "tensorrt"
	openvino = "openvino"


def removeReadOnly(_, name, __):
	os.chmod(name, stat.S_IWRITE)
	os.remove(name)


def markPackages(path):
	for dirpath, dirnames, filenames in os.walk(path):
		if "__init__.py" not in filenames:
			init(dirpath)


def init(dirpath):
	initfile = os.path.join(dirpath, "__init__.py")

	with open(initfile, "w", encoding="utf-8"):
		pass

	return initfile


def pathToPackageName(path, withLibname=True):
	i = 1 if withLibname else 0

	package = path.split(os.path.sep)
	idx = list(reversed(package)).index(libname)

	return ".".join(package[-idx - i:])


class InstallCommand(install):
	description = "command for installation with the ability to select the desired backends and converters"

	user_options = install.user_options + [
		(
			"backend=", None,
			"desired backend. Possible entries: cuda, hip, intel; through comma if several"
		),
		(
			"converter=", None,
			"desired converter which will be included. Possible entries: tensorrt, openvino; through comma if both"
		),
		(
			"no-runtime-check", None,
			"omit runtime check for backends"
		)
	]


	backend, converter = "", ""
	no_runtime_check = 0

	projectPath = os.path.dirname(os.path.abspath(__file__))
	cachePath = os.path.join(projectPath, libname)


	def run(self):
		backends = self.backend.split(",") if len(self.backend) > 0 else []
		converters = self.converter.split(",") if len(self.converter) > 0 else []

		print("backends chosen: %s" % backends)
		print("converters chosen: %s" % converters)

		options = set()
		for option in backends + converters:
			try:
				option = Options[option]
			except KeyError:
				raise ValueError("Invalid option: %s" % option)

			options.add(option)

		handlers = [
			("Compiler", self.installCompilerPackage),

			("CPU", self.installPythonPackage),
			("Intel", self.installIntelPackage),
			(("Cuda", "Hip"), self.installGpuPackages),
			("Converter", self.installConverterPackage),

			("Backend", self.installPythonPackage),
			("Modules", self.installPythonPackage),
			("Containers", self.installPythonPackage),
			("Cost", self.installPythonPackage),
			("Optimizers", self.installPythonPackage),
			("Handlers", self.installPythonPackage),
			("Passes", self.installPythonPackage),
			("Models", self.installPythonPackage),

			("Datasets", self.installPythonPackage),
			("Transformers", self.installPythonPackage),

			("TestData", self.installDataPackage),
			("TestLib", self.installPythonPackage)
		]

		os.mkdir(self.cachePath)

		try:
			self.distribution.package_data = self.installPackages(self.projectPath, self.cachePath, handlers, options)
			markPackages(self.cachePath)

			self.distribution.packages = [libname] + [
				"%s." % libname + pkg for pkg in find_packages(where=self.cachePath)
			]
			super().run()

		finally:
			shutil.rmtree(self.cachePath, onerror=removeReadOnly)


	@staticmethod
	def installPackages(src, dst, handlers, options):
		packageData = {}

		if not os.path.exists(dst):
			os.mkdir(dst)

		for file in os.listdir(src):
			if file.endswith(".py") and os.path.abspath(os.path.join(src, file)) != __file__:
				shutil.copy(os.path.join(src, file), os.path.join(dst, file))

		for name, handler in handlers:
			if isinstance(name, str):
				pkgSrc, pkgDst = os.path.join(src, name), os.path.join(dst, name)
			else:
				pkgSrc, pkgDst = [os.path.join(src, nm) for nm in name], [os.path.join(dst, nm) for nm in name]

			packageData.update(handler(pkgSrc, pkgDst, options))

		return packageData


	@staticmethod
	def installPythonPackage(src, dst, _):
		def ignore(s, names):
			files = {name for name in names if not os.path.isdir(os.path.join(s, name)) and not name.endswith(".py")}
			files.add("__pycache__")

			return files

		shutil.copytree(src, dst, ignore=ignore)
		return {}


	@staticmethod
	def installCompilerPackage(src, dst, _):
		shutil.copytree(src, dst, ignore=lambda s, names: {"__pycache__", "TestData"})
		os.mkdir(os.path.join(dst, "TestData"))

		data = {}

		for dirpath, dirnames, filenames in os.walk(dst):
			buildFiles = [file for file in filenames if any(file.endswith(ext) for ext in [".c", ".h"])]
			if len(buildFiles) > 0:
				data[pathToPackageName(dirpath)] = buildFiles

		return data


	def installGpuPackages(self, src, dst, options):
		data = {}

		cudaSrc, hipSrc = src
		cudaDst, hipDst = dst

		cuda, hip = Options.cuda in options, Options.hip in options

		if cuda or hip:
			shutil.copytree(cudaSrc, cudaDst, ignore=lambda s, names: {"__pycache__", ".gitignore"})

		if cuda:
			from PuzzleLib.Cuda.CheckInstall import checkCudaInstall
			from PuzzleLib.Cuda.Source.Build import main as buildDriver

			data.update(self.installGpuPackage("Cuda", checkCudaInstall, buildDriver, cudaDst))

		if hip:
			shutil.copytree(hipSrc, hipDst, ignore=lambda s, names: {"__pycache__", ".gitignore"})

			from PuzzleLib.Hip.CheckInstall import main as checkHipInstall
			from PuzzleLib.Hip.Source.Build import main as buildDriver

			data.update(self.installGpuPackage("Hip", checkHipInstall, buildDriver, hipDst))

		if cuda or hip:
			shutil.rmtree(os.path.join(cudaDst, "Source"), onerror=removeReadOnly)

		if hip:
			shutil.rmtree(os.path.join(hipDst, "Source"), onerror=removeReadOnly)

		return data


	def installGpuPackage(self, name, checkInstall, buildDriver, dst):
		print("\nChecking if all dependencies for %s are satisfied ..." % name)
		checkInstall(withRuntime=not self.no_runtime_check, withPip=False)

		cwd = os.getcwd()
		try:
			print("\nBuilding %s driver ..." % name)

			os.chdir(os.path.join(dst, "Source"))
			driver = os.path.abspath(buildDriver())

		finally:
			os.chdir(cwd)

		return {pathToPackageName(dst): [os.path.basename(driver)]}


	@staticmethod
	def installIntelPackage(src, dst, options):
		if Options.intel not in options:
			return {}

		shutil.copytree(src, dst, ignore=lambda s, names: {"__pycache__", ".gitignore"})
		data = {}

		from PuzzleLib.Intel.ThirdParty.finddnnl import findDNNL

		print("\nChecking dnnl installation ...")
		lib = findDNNL()

		if os.path.commonpath([dst, lib]) == dst:
			data = {pathToPackageName(dst): [lib]}

		return data


	def installConverterPackage(self, src, dst, options):
		handlers = [
			("Caffe", self.installPythonPackage),
			("MXNet", self.installPythonPackage),

			("Examples", self.installPythonPackage),
			("ONNX", self.installPythonPackage),

			("TensorRT", self.installTensorRTPackage),
			("OpenVINO", self.installOpenVINOPackage)
		]

		data = self.installPackages(src, dst, handlers, options)
		os.mkdir(os.path.join(dst, "TestData"))

		return data


	def installTensorRTPackage(self, src, dst, options):
		if Options.tensorrt not in options:
			return {}

		os.mkdir(dst)
		shutil.copytree(os.path.join(src, "Source"), os.path.join(dst, "Source"))

		from PuzzleLib.Converter.TensorRT.Source.Build import main as buildDriver
		return self.installInferenceEnginePackage("TensorRT", buildDriver, src, dst)


	def installOpenVINOPackage(self, src, dst, options):
		if Options.openvino not in options:
			return {}

		os.mkdir(dst)
		shutil.copytree(os.path.join(src, "Source"), os.path.join(dst, "Source"))

		from PuzzleLib.Converter.OpenVINO.Source.Build import main as buildDriver
		return self.installInferenceEnginePackage("OpenVINO", buildDriver, src, dst)


	@staticmethod
	def installInferenceEnginePackage(name, buildDriver, src, dst):
		for file in os.listdir(src):
			if file.endswith(".py"):
				shutil.copy(os.path.join(src, file), os.path.join(dst, file))

		shutil.copytree(os.path.join(src, "Tests"), os.path.join(dst, "Tests"))
		os.mkdir(os.path.join(dst, "TestData"))

		cwd = os.getcwd()
		try:
			print("\nBuilding %s driver ..." % name)

			sourcePath = os.path.join(dst, "Source")
			os.chdir(sourcePath)

			driver = buildDriver()

		finally:
			os.chdir(cwd)

		shutil.rmtree(sourcePath, onerror=removeReadOnly)
		return {pathToPackageName(dst): [os.path.basename(driver)]}


	@staticmethod
	def installDataPackage(src, dst, _):
		os.mkdir(dst)

		data = ["test.tar", "test.zip"]
		for file in data:
			shutil.copy(os.path.join(src, file), os.path.join(dst, file))

		return {pathToPackageName(dst): data}


class SdistCommand(sdist):
	projectPath = os.path.dirname(os.path.abspath(__file__))


	def run(self):
		initfiles = []

		handlers = [
			("Compiler", self.distributeCompilerPackage),

			("CPU", self.distributePythonPackage),
			("Intel", self.distributePythonPackage),
			("Hip", self.distributeGpuPackage),
			("Cuda", self.distributeGpuPackage),
			("Converter", self.distributeConverterPackage),

			("Backend", self.distributePythonPackage),
			("Modules", self.distributePythonPackage),
			("Containers", self.distributePythonPackage),
			("Cost", self.distributePythonPackage),
			("Optimizers", self.distributePythonPackage),
			("Handlers", self.distributePythonPackage),
			("Passes", self.distributePythonPackage),
			("Models", self.distributePythonPackage),

			("Datasets", self.distributePythonPackage),
			("Transformers", self.distributePythonPackage),

			("TestData", self.distributeDataPackage),
			("TestLib", self.distributePythonPackage)
		]

		try:
			initfiles, data = self.distributePackages(self.projectPath, handlers)

			self.distribution.package_data = data
			self.distribution.packages = find_packages(where=self.projectPath)

			super().run()

		finally:
			for initfile in initfiles:
				os.unlink(initfile)


	@staticmethod
	def distributePackage(path, exclude, includeExts):
		initfiles = []
		data = {}

		for dirpath, dirnames, filenames in os.walk(path):
			for exdir in exclude:
				if exdir in dirnames:
					dirnames.remove(exdir)

			includeFiles = [file for file in filenames if any(file.endswith(ext) for ext in includeExts)]
			if len(includeFiles) > 0:
				data[pathToPackageName(dirpath, withLibname=False)] = includeFiles

			if "__init__.py" not in filenames:
				initfiles.append(init(dirpath))

		return initfiles, data


	@staticmethod
	def distributePackages(path, handlers, initPath=False):
		initfiles = []
		data = {}

		for name, handler in handlers:
			pkgInitfiles, pkgData = handler(os.path.join(path, name))

			data.update(pkgData)
			initfiles.extend(pkgInitfiles)

		if initPath:
			initfiles.append(init(path))

		return initfiles, data


	def distributePythonPackage(self, path):
		return self.distributePackage(path=path, exclude=["__pycache__"], includeExts=[])


	def distributeCompilerPackage(self, path):
		return self.distributePackage(path=path, exclude=["__pycache__"], includeExts=[".c", ".h"])


	def distributeGpuPackage(self, path):
		return self.distributeCompilerPackage(path)


	def distributeConverterPackage(self, path):
		handlers = [
			("Caffe", self.distributePythonPackage),
			("MXNet", self.distributePythonPackage),

			("Examples", self.distributePythonPackage),
			("ONNX", self.distributePythonPackage),

			("TensorRT", self.distributeTensorRTPackage),
			("OpenVINO", self.distributeOpenVINOPackage)
		]

		return self.distributePackages(path, handlers, True)


	def distributeTensorRTPackage(self, path):
		return self.distributePackage(path=path, exclude=["__pycache__"], includeExts=[".cpp", ".h", ".cu"])


	def distributeOpenVINOPackage(self, path):
		return self.distributePackage(path=path, exclude=["__pycache__"], includeExts=[".cpp"])


	def distributeDataPackage(self, path):
		data = ["test.tar", "test.zip"]

		initfiles, pkgData = self.distributePythonPackage(path)
		pkgData.update({pathToPackageName(path, withLibname=False): data})

		return initfiles, pkgData


def main():
	setup(
		name=libname,
		version=version,
		cmdclass={
			"install": InstallCommand,
			"sdist": SdistCommand
		},
		url="https://puzzlelib.org",
		download_url="https://github.com/puzzlelib/PuzzleLib/tags",
		author="Ashmanov Neural Networks",
		python_requires=">=3.5",
		license="Apache-2.0",
		keywords=["puzzlelib", "deep learning", "neural nets"]
	)


if __name__ == "__main__":
	main()
