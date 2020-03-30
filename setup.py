import os, stat, shutil, subprocess
from enum import Enum

from setuptools import setup, find_packages
from setuptools.command.install import install


libname = "PuzzleLib"


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
			initfile = os.path.join(dirpath, "__init__.py")

			with open(initfile, "w", encoding="utf-8"):
				pass


def pathToPackageName(path):
	package = path.split(os.path.sep)
	idx = list(reversed(package)).index(libname)

	return ".".join(package[-idx - 1:])


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
		)
	]


	backend, converter = "", ""

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
			("Compiler", self.handleCompilerPackage),

			("CPU", self.handlePythonPackage),
			("Intel", self.handleIntelPackage),
			(("Cuda", "Hip"), self.handleGpuPackages),
			("Converter", self.handleConverterPackage),

			("Backend", self.handlePythonPackage),
			("Modules", self.handlePythonPackage),
			("Containers", self.handlePythonPackage),
			("Cost", self.handlePythonPackage),
			("Optimizers", self.handlePythonPackage),
			("Handlers", self.handlePythonPackage),
			("Passes", self.handlePythonPackage),
			("Models", self.handlePythonPackage),

			("Datasets", self.handlePythonPackage),
			("Transformers", self.handlePythonPackage),

			("TestData", self.handleDataPackage),
			("TestLib", self.handlePythonPackage)
		]

		os.mkdir(self.cachePath)

		try:
			self.distribution.package_data = self.handlePackages(self.projectPath, self.cachePath, handlers, options)
			markPackages(self.cachePath)

			self.distribution.packages = [libname] + [
				"%s." % libname + pkg for pkg in find_packages(where=self.cachePath)
			]
			super().run()

		finally:
			shutil.rmtree(self.cachePath, onerror=removeReadOnly)


	@staticmethod
	def handlePackages(src, dst, handlers, options):
		packageData = {}

		for file in os.listdir(src):
			if file.endswith(".py") and os.path.abspath(os.path.join(src, file)) != __file__:
				shutil.copy(file, os.path.join(dst, file))

		for name, handler in handlers:
			if isinstance(name, str):
				pkgSrc, pkgDst = os.path.join(src, name), os.path.join(dst, name)
			else:
				pkgSrc, pkgDst = [os.path.join(src, nm) for nm in name], [os.path.join(dst, nm) for nm in name]

			packageData.update(handler(pkgSrc, pkgDst, options))

		return packageData


	@staticmethod
	def handlePythonPackage(src, dst, _):
		def ignore(s, names):
			files = {name for name in names if not os.path.isdir(os.path.join(s, name)) and not name.endswith(".py")}
			files.add("__pycache__")

			return files

		shutil.copytree(src, dst, ignore=ignore)
		return {}


	@staticmethod
	def handleCompilerPackage(src, dst, _):
		shutil.copytree(src, dst, ignore=lambda s, names: {"__pycache__", "TestData"})
		os.mkdir(os.path.join(dst, "TestData"))

		data = {}

		for dirpath, dirnames, filenames in os.walk(dst):
			buildFiles = [file for file in filenames if any(file.endswith(ext) for ext in [".c", ".h"])]
			if len(buildFiles) > 0:
				data[pathToPackageName(dirpath)] = buildFiles

		return data


	def handleGpuPackages(self, src, dst, options):
		data = {}

		cudaSrc, hipSrc = src
		cudaDst, hipDst = dst

		if Options.cuda in options:
			shutil.copytree(cudaSrc, cudaDst, ignore=lambda s, names: {"__pycache__", ".gitignore"})

			from PuzzleLib.Cuda.CheckInstall import main as installChecker
			from PuzzleLib.Cuda.Source.Build import main as driverBuilder

			data.update(self.handleGpuPackage("Cuda", installChecker, driverBuilder, cudaDst))

		if Options.hip in options:
			shutil.copytree(hipSrc, hipDst, ignore=lambda s, names: {"__pycache__", ".gitignore"})

			from PuzzleLib.Hip.CheckInstall import main as installChecker
			from PuzzleLib.Hip.Source.Build import main as driverBuilder

			data.update(self.handleGpuPackage("Hip", installChecker, driverBuilder, hipDst))

		if Options.cuda in options:
			shutil.rmtree(os.path.join(cudaDst, "Source"), onerror=removeReadOnly)

		if Options.hip in options:
			shutil.rmtree(os.path.join(hipDst, "Source"), onerror=removeReadOnly)

		return data


	@staticmethod
	def handleGpuPackage(name, installChecker, driverBuilder, dst):
		print("\nChecking if all dependencies for %s are satisfied ..." % name)
		installChecker()

		cwd = os.getcwd()
		try:
			print("\nBuilding %s driver ..." % name)

			os.chdir(os.path.join(dst, "Source"))
			driver = os.path.abspath(driverBuilder())

		finally:
			os.chdir(cwd)

		return {pathToPackageName(dst): [os.path.basename(driver)]}


	@staticmethod
	def handleIntelPackage(src, dst, options):
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


	def handleConverterPackage(self, src, dst, options):
		handlers = [
			("Caffe", self.handlePythonPackage),
			("MXNet", self.handlePythonPackage),

			("Examples", self.handlePythonPackage),
			("ONNX", self.handlePythonPackage),

			("TensorRT", self.handleTensorRTPackage),
			("OpenVINO", self.handleOpenVINOPackage)
		]

		data = self.handlePackages(src, dst, handlers, options)
		os.mkdir(os.path.join(dst, "TestData"))

		return data


	def handleTensorRTPackage(self, src, dst, options):
		if Options.tensorrt not in options:
			return {}

		os.mkdir(dst)
		shutil.copytree(os.path.join(src, "Source"), os.path.join(dst, "Source"))

		from PuzzleLib.Converter.TensorRT.Source.Build import main as driverBuilder
		return self.handleInferenceEnginePackage("TensorRT", driverBuilder, src, dst)


	def handleOpenVINOPackage(self, src, dst, options):
		if Options.openvino not in options:
			return {}

		os.mkdir(dst)
		shutil.copytree(os.path.join(src, "Source"), os.path.join(dst, "Source"))

		from PuzzleLib.Converter.OpenVINO.Source.Build import main as driverBuilder
		return self.handleInferenceEnginePackage("OpenVINO", driverBuilder, src, dst)


	@staticmethod
	def handleInferenceEnginePackage(name, driverBuilder, src, dst):
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

			driver = driverBuilder()

		finally:
			os.chdir(cwd)

		shutil.rmtree(sourcePath, onerror=removeReadOnly)
		return {pathToPackageName(dst): [os.path.basename(driver)]}


	@staticmethod
	def handleDataPackage(src, dst, __):
		os.mkdir(dst)

		data = ["test.tar", "test.zip"]
		for file in data:
			shutil.copy(os.path.join(src, file), os.path.join(dst, file))

		return {pathToPackageName(dst): data}


def main():
	setup(
		name=libname,
		version="1.0.0",
		cmdclass={"install": InstallCommand},
		url="https://puzzlelib.org",
		python_requires=">=3.5",
		license="Apache-2.0"
	)


if __name__ == "__main__":
	main()
