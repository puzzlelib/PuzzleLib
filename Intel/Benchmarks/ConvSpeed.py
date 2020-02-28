from PuzzleLib.Intel.Wrappers import DNNL


def main():
	datashape = (16, 32, 64, 64)
	Wshape = (64, 32, 3, 3)

	stride, pad = 1, 0
	timeConv(datashape, Wshape, stride, pad)


def timeConv(datashape, Wshape, stride, pad):
	fwdResults, bwdFilterResults, bwdDataResults = DNNL.convNdbenchmark(datashape, Wshape, stride, pad)

	formatstr = "%-40s %-25s %-28s"

	print("Forward results:")
	for res in fwdResults:
		print(formatstr % (
			"Algo %s" % res.algo, "time %.6f secs" % res.time, "memory %.6f mbytes" % (res.memory / 1024**2)
		))

	print("\nBackward filter results:")
	for res in bwdFilterResults:
		print(formatstr % (
			"Algo %s" % res.algo, "time %.6f secs" % res.time, "memory %.6f mbytes" % (res.memory / 1024**2)
		))

	print("\nBackward data results:")
	for res in bwdDataResults:
		print(formatstr % (
			"Algo %s" % DNNL.ConvAlgo(res.algo), "time %.6f secs" % res.time,
			"memory %.6f mbytes" % (res.memory / 1024**2)
		))


if __name__ == "__main__":
	main()
