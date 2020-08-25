import sys, os, subprocess, tempfile
from colorama import Fore, Style


if "PYCHARM_HOSTED" not in os.environ:
	import colorama
	colorama.init()


cudaTestKernel = """

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


def checkInstall(name, compiler, download, envpath):
	print("%sChecking %s installation ...%s" % (Fore.LIGHTBLUE_EX, name, Style.RESET_ALL))

	try:
		version = subprocess.getoutput("%s --version" % compiler).split()[-1]

	except Exception as e:
		error = "%s%s is not found with error(s):%s\n%s" % (Fore.RED, name, Style.RESET_ALL, e)
		note = "Download and install appropriate version from %s" % download
		raise RuntimeError("%s\n%s" % (error, note))

	print("%s%s %s and SDK libraries are found!%s" % (Fore.LIGHTGREEN_EX, name, version, Style.RESET_ALL))
	print("Continuing ...", end="\n\n")

	if sys.platform != "win32":
		return

	print("%sChecking %s environment on Windows platform ...%s" % (Fore.LIGHTBLUE_EX, name, Style.RESET_ALL))
	RUNTIME_PATH = os.environ.get(envpath, None)

	if RUNTIME_PATH is None:
		raise RuntimeError(
			"%s%s is not set - set it to CUDA installation path!%s" % (Fore.RED, envpath, Style.RESET_ALL)
		)

	print("%s%s is set!%s" % (Fore.LIGHTGREEN_EX, envpath, Style.RESET_ALL))
	print("Continuing ...", end="\n\n")


def checkRuntime(name, compiler, kernel, ext):
	print("%sChecking %s compiler ...%s" % (Fore.LIGHTBLUE_EX, compiler.upper(), Style.RESET_ALL))

	temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=ext, delete=False)
	exefile = os.path.join(os.path.dirname(temp.name), "a.out")

	try:
		with temp:
			temp.write(kernel)

		try:
			res = subprocess.check_output([compiler, "-o", exefile, temp.name])
			print("%s%s compiled test kernel with output:%s %s" % (
				Fore.LIGHTGREEN_EX, compiler, Style.RESET_ALL, res.decode("utf-8")
			))

			print("Continuing ...", end="\n\n")

		except subprocess.CalledProcessError as e:
			raise RuntimeError("%s%s failed compiling test kernel with error(s):%s\n%s" % (
				Fore.RED, compiler, Style.RESET_ALL, e.output.decode("utf-8")
			))

	finally:
		os.remove(temp.name)

	print("%sChecking compiled %s kernel ...%s" % (Fore.LIGHTBLUE_EX, name, Style.RESET_ALL))

	try:
		result = subprocess.check_output(exefile, stderr=subprocess.PIPE).decode("utf-8")
		print(
			"%sTest kernel answered:%s %s\nContinuing ..." % (Fore.LIGHTGREEN_EX, Style.RESET_ALL, result), end="\n\n"
		)

	except subprocess.CalledProcessError as e:
		raise RuntimeError(
			"%sTest kernel failed with error:%s %s" % (Fore.RED, Style.RESET_ALL, e.stderr.decode("utf-8"))
		)

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

	installed = subprocess.check_output([pip, "list", "--format=freeze"]).decode("utf-8")
	installed = {k: v for k, v in map(lambda s: s.split(sep="=="), installed.splitlines())}

	for package in packages:
		print("%sChecking %s installation ...%s" % (Fore.LIGHTBLUE_EX, package, Style.RESET_ALL))
		version = installed.get(package, None)

		if version is None:
			print("%s%s is not installed%s\n" % (Fore.YELLOW, package, Style.RESET_ALL))

			try:
				print("%sInstalling %s ...%s" % (Fore.LIGHTBLUE_EX, package, Style.RESET_ALL))
				cmd = [pip, "install"]

				if sys.platform != "win32":
					cmd.append("--user")

				result = subprocess.check_output(cmd + [package])
				print(result.decode("utf-8"))

			except subprocess.CalledProcessError as e:
				raise RuntimeError(
					"%s%s installation error:%s\n%s" % (Fore.RED, package, Style.RESET_ALL, e.output.decode("utf-8"))
				)

		else:
			print("%sFound package %s==%s%s" % (Fore.LIGHTGREEN_EX, package, version, Style.RESET_ALL))

		print("Continuing ...", end="\n\n")


def checkCudaInstall(withRuntime, withPip):
	checkInstall(
		name="CUDA", compiler="nvcc", download="https://developer.nvidia.com/cuda-downloads", envpath="CUDA_PATH"
	)

	if withRuntime:
		checkRuntime(name="CUDA", compiler="nvcc", kernel=cudaTestKernel, ext=".cu")

	if withPip:
		checkPipPackages()

	print("%sAll done, exiting ...%s" % (Fore.LIGHTGREEN_EX, Style.RESET_ALL))


def main():
	try:
		checkCudaInstall(withRuntime=True, withPip=True)

	except RuntimeError as e:
		print(e)

		print("Exiting ...")
		sys.exit(1)


if __name__ == "__main__":
	main()
