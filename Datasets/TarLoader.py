import tarfile

from PuzzleLib.Datasets.InputLoader import InputLoader


class TarLoader(InputLoader):
	def checkInput(self, archivename):
		if not tarfile.is_tarfile(archivename):
			raise RuntimeError("'%s' is not tar file" % archivename)


	def openInput(self, archivename):
		return tarfile.open(archivename)


	def loadFilelist(self, archive):
		return [file for file in archive.getnames() if any([file.lower().endswith(ext) for ext in self.exts])]


	def openFile(self, archive, file):
		return archive.extractfile(file)


def unittest():
	loader = TarLoader()
	loader.load("../TestData/test.tar", maxsamples=5, filepacksize=3)
	loader.clear()


if __name__ == "__main__":
	unittest()
