from PuzzleLib.Cuda.Source.Build import prepareCompiler, generateTemplates, collectCoreSources, collectLibSources


def buildDriver(debugmode, verbose):
	cc = prepareCompiler(debugmode, verbose)
	prepareHip(cc)

	generateTemplates(path="../../Cuda/Source")

	driver = "../Driver" + cc.pydext
	cc.build(driver, collectSources(path="../../Cuda/Source")).clearPath("..")

	return driver


def prepareHip(cc):
	cc.cppMode(True).addDefine("__HIP_PLATFORM_HCC__")
	cc.cflags.extend(["-x", "c++"])

	cc.addLibrary(
		"hip",
		[
			".", "/opt/rocm/hsa/include", "/opt/rocm/hip/include",
			"/opt/rocm/hiprand/include", "/opt/rocm/rocrand/include",
			"/opt/rocm/rocblas/include", "/opt/rocm/miopen/include"
		],
		["/opt/rocm/hip/lib", "/opt/rocm/hiprand/lib", "/opt/rocm/rocblas/lib", "/opt/rocm/miopen/lib"],
		["hip_hcc", "hiprtc", "hiprand", "rocblas"]
	)


def collectSources(path):
	return collectCoreSources(path) + collectLibSources(path)


def main():
	return buildDriver(debugmode=0, verbose=2)


if __name__ == "__main__":
	main()
