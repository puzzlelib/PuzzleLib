import zipfile

from PuzzleLib.Datasets.InputLoader import InputLoader


class ZipLoader(InputLoader):
	def checkInput(self, archivename):
		if not zipfile.is_zipfile(archivename):
			raise RuntimeError("'%s' is not zip file" % archivename)


	def openInput(self, archivename):
		return zipfile.ZipFile(archivename)


	def loadFilelist(self, archive):
		return [file for file in archive.namelist() if any([file.lower().endswith(ext) for ext in self.exts])]


	def openFile(self, archive, file):
		return archive.open(file)


def unittest():
	loader = ZipLoader()
	loader.load("../TestData/test.zip", maxsamples=5, filepacksize=3)
	loader.clear()


if __name__ == "__main__":
	unittest()
