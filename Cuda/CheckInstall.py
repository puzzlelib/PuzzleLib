import sys, os, subprocess
import tempfile

from colorama import Fore, Style


if "PYCHARM_HOSTED" not in os.environ:
	import colorama
	colorama.init()


testKernel = """

#include <stdio.h>


__global__ void iaxpy(int *y, const int *x, int a, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) y[i] += a * x[i];
}


#define CUDA_ASSERT(status) do { if (!cudaAssertStatus((status), __LINE__)) exit(1); } while (0)
inline bool cudaAssertStatus(cudaError_t code, int line)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr, "%s (line:%d)\\n", cudaGetErrorString(code), line);
		return false;
	}

	return true;
}


int main()
{
	int exitcode = 0;

	const int SIZE = 1 << 20;
	const int NBYTES = SIZE * sizeof(int);

	int *hostx = (int *)malloc(NBYTES);
	int *hosty = (int *)malloc(NBYTES);

	int *devx = NULL, *devy = NULL;
	CUDA_ASSERT(cudaMalloc(&devx, NBYTES));
	CUDA_ASSERT(cudaMalloc(&devy, NBYTES));

	for (int i = 0; i < SIZE; i++)
	{
		hostx[i] = i;
		hosty[i] = -i * 2;
	}

	CUDA_ASSERT(cudaMemcpy(devx, hostx, NBYTES, cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(devy, hosty, NBYTES, cudaMemcpyHostToDevice));

	const int NT = 256;
	iaxpy<<<(SIZE + NT - 1) / NT, NT>>>(devy, devx, 2, SIZE);

	CUDA_ASSERT(cudaMemcpy(hosty, devy, NBYTES, cudaMemcpyDeviceToHost));

	CUDA_ASSERT(cudaFree(devx));
	CUDA_ASSERT(cudaFree(devy));

	for (int i = 0; i < SIZE; i++)
		if (hosty[i] != 0)
		{
			fprintf(stderr, "kernel invocation failed!");

			exitcode = 1;
			goto exit;
		}

	printf("finished successfully!");
	fflush(stdout);

exit:
	free(hostx);
	free(hosty);

	return exitcode;
}

"""


def main():
	checkCuda()
	checkNVCC()
	checkPipPackages()

	print("%sAll done, exiting ...%s" % (Fore.LIGHTGREEN_EX, Style.RESET_ALL))


def checkCuda():
	print("%sChecking CUDA installation ...%s" % (Fore.LIGHTBLUE_EX, Style.RESET_ALL))

	try:
		version = subprocess.getoutput("nvcc --version").split()[-1]

	except Exception as e:
		print("%sCUDA library is not found with error:%s\n%s" % (Fore.RED, Style.RESET_ALL, e))
		print("Download and install appropriate version from https://developer.nvidia.com/cuda-downloads")

		print("Exiting ...")
		sys.exit(1)

	print("%sCUDA %s and SDK libraries are found!%s" % (Fore.LIGHTGREEN_EX, version, Style.RESET_ALL))
	print("Continuing ...", end="\n\n")

	if sys.platform != "win32":
		return

	print("%sChecking CUDA environment on Windows platform ...%s" % (Fore.LIGHTBLUE_EX, Style.RESET_ALL))
	CUDA_PATH = os.environ.get("CUDA_PATH", None)

	if CUDA_PATH is None:
		print("%sCUDA_PATH is not set - set it to CUDA installation path!%s" % (Fore.RED, Style.RESET_ALL))

		print("Exiting ...")
		sys.exit(1)

	print("%sCUDA_PATH is set!%s" % (Fore.LIGHTGREEN_EX, Style.RESET_ALL))
	print("Continuing ...", end="\n\n")


def checkNVCC():
	print("%sChecking NVCC compiler ...%s" % (Fore.LIGHTBLUE_EX, Style.RESET_ALL))

	temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".cu", delete=False)
	exefile = os.path.join(os.path.dirname(temp.name), "a.out")

	try:
		with temp:
			temp.write(testKernel)

		try:
			res = subprocess.check_output(["nvcc", "-o", exefile, temp.name])
			print("%snvcc compiled test kernel with output:%s %s" % (
				Fore.LIGHTGREEN_EX, Style.RESET_ALL, res.decode("utf-8")
			))

			print("Continuing ...", end="\n\n")

		except subprocess.CalledProcessError as e:
			print("%snvcc failed compiling test kernel with error:%s\n%s" % (
				Fore.RED, Style.RESET_ALL, e.output.decode("utf-8")
			))

			print("Exiting ...")
			sys.exit(1)

	finally:
		os.remove(temp.name)

	print("%sChecking compiled CUDA kernel ...%s" % (Fore.LIGHTBLUE_EX, Style.RESET_ALL))

	try:
		result = subprocess.check_output(exefile, stderr=subprocess.PIPE).decode("utf-8")
		print(
			"%sTest kernel answered:%s %s\nContinuing ..." % (Fore.LIGHTGREEN_EX, Style.RESET_ALL, result), end="\n\n"
		)

	except subprocess.CalledProcessError as e:
		print("%sTest kernel failed with error:%s %s" % (Fore.RED, Style.RESET_ALL, e.stderr.decode("utf-8")))

		print("Exiting ...")
		sys.exit(1)

	finally:
		os.remove(exefile)


def checkPipPackages():
	print("%sChecking python packages ...%s\n" % (Fore.LIGHTBLUE_EX, Style.RESET_ALL))
	packages = ["numpy", "h5py", "Pillow", "graphviz", "colorama"]

	try:
		pip = "pip3"
		subprocess.check_output([pip])

	except subprocess.CalledProcessError:
		pip = "pip"

	installed = subprocess.check_output([pip, "list", "freeze"]).decode("utf-8")
	installed = {k: v for k, v in map(lambda s: s.split(), installed.splitlines())}

	for package in packages:
		print("%sChecking package '%s' installation ...%s" % (Fore.LIGHTBLUE_EX, package, Style.RESET_ALL))
		version = installed.get(package, None)

		if version is None:
			print("%sPackage '%s' is not installed%s\n" % (Fore.YELLOW, package, Style.RESET_ALL))

			try:
				print("%sInstalling package %s ...%s" % (Fore.LIGHTBLUE_EX, package, Style.RESET_ALL))
				cmd = [pip, "install"]

				if sys.platform != "win32":
					cmd.append("--user")

				result = subprocess.check_output(cmd + [package])
				print(result.decode("utf-8"))

			except subprocess.CalledProcessError as e:
				print("%sPackage '%s' installation error:%s\n%s" % (
					Fore.RED, package, Style.RESET_ALL, e.output.decode("utf-8")
				))

				print("Exiting ...")
				sys.exit(1)

		else:
			print("%sFound package '%s' == %s%s" % (Fore.LIGHTGREEN_EX, package, version, Style.RESET_ALL))

		print("Continuing ...", end="\n\n")


if __name__ == "__main__":
	main()
