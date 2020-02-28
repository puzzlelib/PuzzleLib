import os, atexit, subprocess
from setuptools import setup, find_packages

from PuzzleLib.Config import libname


def main():
	ignoredPaths = [".git", "__pycache__", "TestData", "SetupTools"]
	toRemove, packageData = [], {}

	for dirpath, dirnames, filenames in os.walk(".."):
		for dirname in dirnames:
			if any(dirname.endswith(path) for path in ignoredPaths):
				dirnames.remove(dirname)
				print("Removed %s from wheel" % os.path.abspath(os.path.join(dirpath, dirname)))

		if not "__init__.py" in filenames:
			initfile = os.path.join(dirpath, "__init__.py")

			open(initfile, "w", encoding="utf-8")
			toRemove.append(initfile)

		for filename in filenames:
			if os.path.splitext(filename)[-1] in {".h", ".dll"}:
				package = libname + dirpath[2:].replace(os.path.sep, ".")

				lst = packageData.get(package, [])
				lst.append(filename)

				packageData[package] = lst

	print("Including package external files: %s" % packageData)

	def cleanInitFiles():
		os.chdir(os.getcwd())

		for file in toRemove:
			os.remove(file)

	atexit.register(cleanInitFiles)

	label = subprocess.check_output(["git", "describe", "--always"]).decode().strip()
	os.chdir("../..")

	setup(
		name=libname,
		version="1.0.%s" % label,

		packages=find_packages(exclude=["PuzzleLib.SetupTools"]),
		package_data=packageData
	)


if __name__ == "__main__":
	main()
