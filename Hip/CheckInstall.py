from PuzzleLib.Cuda.CheckInstall import checkRuntime, checkCompiler, checkPipPackages


hipTestKernel = """

#include <stdio.h>
#include <hip/hip_runtime.h>


__global__ void iaxpy(int *y, const int *x, int a, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) y[i] += a * x[i];
}


#define HIP_ASSERT(status) do { if (!hipAssertStatus((status), __LINE__)) exit(1); } while (0)
inline bool hipAssertStatus(hipError_t code, int line)
{
	if (code != hipSuccess) 
	{
		fprintf(stderr, "%s (line:%d)\\n", hipGetErrorString(code), line);
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
	HIP_ASSERT(hipMalloc(&devx, NBYTES));
	HIP_ASSERT(hipMalloc(&devy, NBYTES));

	for (int i = 0; i < SIZE; i++)
	{
		hostx[i] = i;
		hosty[i] = -i * 2;
	}

	HIP_ASSERT(hipMemcpy(devx, hostx, NBYTES, hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(devy, hosty, NBYTES, hipMemcpyHostToDevice));

	const int NT = 256;
	hipLaunchKernelGGL(iaxpy, dim3((SIZE + NT - 1) / NT), dim3(NT), 0, 0, devy, devx, 2, SIZE);

	HIP_ASSERT(hipMemcpy(hosty, devy, NBYTES, hipMemcpyDeviceToHost));

	HIP_ASSERT(hipFree(devx));
	HIP_ASSERT(hipFree(devy));

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
	checkRuntime(
		name="HIP", compiler="hipcc",
		download="https://rocm.github.io/install.html#ubuntu-support---installing-from-a-debian-repository",
		envpath="HIP_PATH"
	)
	checkCompiler(name="HIP", compiler="hipcc", kernel=hipTestKernel, ext=".hip.cpp")
	checkPipPackages()


if __name__ == "__main__":
	main()
