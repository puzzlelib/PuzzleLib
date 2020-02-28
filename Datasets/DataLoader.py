import os


class DataLoader:
	def __init__(self, datanames=None, cachename=None):
		self.cachename = cachename

		if datanames is None:
			self.datanames = ["data"]
		else:
			if isinstance(datanames, list) or isinstance(datanames, tuple):
				self.datanames = datanames
			else:
				self.datanames = [datanames]


	def clear(self):
		if os.path.exists(self.cachename):
			os.remove(self.cachename)
