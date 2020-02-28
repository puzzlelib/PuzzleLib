from PuzzleLib.OpenCL.Wrappers import MIOpen


def main():
	from PuzzleLib.OpenCL.Utils import autoinit
	autoinit()

	datashape = (128, 32, 64, 64)
	Wshape = (64, 32, 11, 11)

	stride, pad, = 1, 0
	timeConv(datashape, Wshape, stride, pad)


def timeConv(datashape, Wshape, stride, pad):
	fwdResults, bwdFilterResults, bwdDataResults = MIOpen.conv2dbenchmark(datashape, Wshape, stride, pad)

	formatstr = "%-40s %-25s %-28s"

	print("Forward results:")
	for res in fwdResults:
		print(formatstr % ("Algo %s" % MIOpen.ConvFwdAlgo(res.algo), "time %.6f secs" % res.time,
						   "memory %.6f mbytes" % (res.memory / 1024**2)))

	print("\nBackward filter results:")
	for res in bwdFilterResults:
		print(formatstr % ("Algo %s" % MIOpen.ConvBwdFilterAlgo(res.algo), "time %.6f secs" % res.time,
						   "memory %.6f mbytes" % (res.memory / 1024**2)))

	print("\nBackward data results:")
	for res in bwdDataResults:
		print(formatstr % ("Algo %s" % MIOpen.ConvBwdDataAlgo(res.algo), "time %.6f secs" % res.time,
						   "memory %.6f mbytes" % (res.memory / 1024**2)))


if __name__ == "__main__":
	main()
