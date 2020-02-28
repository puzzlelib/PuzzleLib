import os, shutil, zipfile

from PuzzleLib.Datasets.InputLoader import InputLoader


class PathLoader(InputLoader):
	def __init__(self, onFile=None, exts=None, dataname=None, cachename=None, onFileList=None, doOpen=True):
		super().__init__(onFile, exts, dataname, cachename, onFileList)
		self.doOpen = doOpen


	class Path:
		def __init__(self, path):
			self.path = path


		def __enter__(self):
			return self


		def __exit__(self, exc_type, exc_val, exc_tb):
			pass


	def checkInput(self, path):
		if not os.path.exists(path):
			raise RuntimeError("Path '%s' does not exist" % path)


	def openInput(self, path):
		return self.Path(path)


	def loadFilelist(self, path):
		lst = []

		for dirpath, dirnames, filenames in os.walk(path.path):
			lst.extend([file for file in filenames if any([file.lower().endswith(ext) for ext in self.exts])])

		return lst


	def openFile(self, path, file):
		fullname = os.path.join(path.path, file)
		return open(fullname, mode="rb") if self.doOpen else fullname


def unittest():
	zipname = "../TestData/test.zip"
	path = os.path.splitext(zipname)[0]

	zipfile.ZipFile(zipname).extractall(path)

	loader = PathLoader()
	loader.load(path)
	loader.clear()

	shutil.rmtree(path)


if __name__ == "__main__":
	unittest()
