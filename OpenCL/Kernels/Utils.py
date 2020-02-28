warpSize = 64
nthreads = 256


atomicAddTmpl = """

void atomicAddCAS(__global float *location, float value)
{
	float old = *location;
	float sum = old + value;

	while (atomic_cmpxchg((__global int*)location, *((int *)&old), *((int *)&sum)) != *((int*)&old))
	{
		old = *location;
		sum = old + value;
	}
}

"""


def roundUpDiv(a, b):
	return (a + b - 1) // b


def roundUp(a, b):
	return roundUpDiv(a, b) * b
