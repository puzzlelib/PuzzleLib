from random import Random
from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver
from PuzzleLib.OpenCL.Kernels.Templates import create_some_context


randomTmpl = Template("""

#define ROUNDS $rounds
#define TRANSFORM $transform

#define PHILOX_W32_0 ((uint)0x9E3779B9)
#define PHILOX_W32_1 ((uint)0xBB67AE85)

#define PHILOX_M4x32_0 ((uint)0xD2511F53)
#define PHILOX_M4x32_1 ((uint)0xCD9E8D57)


typedef struct
{
	uint v[2];
} array2x32;


typedef struct
{
	uint v[4];
} array4x32;


inline uint mulhilo32(uint a, uint b, uint* hip)
{
	*hip = mul_hi(a, b);
	return a * b;
}


inline array2x32 bumpkey(array2x32 key)
{
	key.v[0] += PHILOX_W32_0;
	key.v[1] += PHILOX_W32_1;
	return key;
}


inline array4x32 philox4x32round(array4x32 ctr, array2x32 key)
{
	uint hi0, hi1;

	uint lo0 = mulhilo32(PHILOX_M4x32_0, ctr.v[0], &hi0);
	uint lo1 = mulhilo32(PHILOX_M4x32_1, ctr.v[2], &hi1);

	array4x32 out = {{hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
	return out;
}


inline array4x32 performRounds(array2x32 key, array4x32 ctr)
{
	if (ROUNDS > 0)  {                     ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 1)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 2)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 3)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 4)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 5)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 6)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 7)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 8)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 9)  { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 10) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 11) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 12) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 13) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 14) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }
	if (ROUNDS > 15) { key = bumpkey(key); ctr = philox4x32round(ctr, key); }

	return ctr;
}


inline uint4 genBits(array2x32 key, array4x32 counter)
{
	union
	{
		uint4 v;
		array4x32 counter;
	} bits;

	bits.counter = performRounds(key, counter);
	return bits.v;
}


inline array4x32 updateCounter(array4x32 counter)
{
	if (++counter.v[0] == 0)
		if (++counter.v[1] == 0)
			++counter.v[2];

	return counter;
}


inline float4 boxMuller(float4 v)
{
	float r0 = sqrt(-2 * log(v.x));
	float c0;
	float s0 = sincos((float)(2 * M_PI) * v.y, &c0); 

	float r1 = sqrt(-2 * log(v.z));
	float c1;
	float s1 = sincos((float)(2 * M_PI) * v.w, &c1);

	return (float4)(r0 * c0, r0 * s0, r1 * c1, r1 * s1);
}


inline float4 convertRandomNum(uint4 gen, float scale, float shift)
{
	return shift + scale * TRANSFORM(((float)2.3283064365386963e-10) * convert_float4(gen));
}


__kernel void philox4x32(__global float *data, long size, int k1, int c0, int c1, int c2, int c3,
						 float scale, float shift)
{
	array2x32 key = {{get_global_id(0), k1}};
	array4x32 counter = {{c0, c1, c2, c3}};

	unsigned long idx = get_global_id(0) * 4;
	while (idx + 4 < size)
	{
		float4 ran = convertRandomNum(genBits(key, counter), scale, shift);
		counter = updateCounter(counter);

		vstore4(ran, 0, &data[idx]);
		idx += 4 * get_global_size(0);
	}

	float4 tail = convertRandomNum(genBits(key, counter), scale, shift);
	if (idx < size)
		data[idx] = tail.x;
	if (idx + 1 < size)
		data[idx + 1] = tail.y;
	if (idx + 2 < size)
		data[idx + 2] = tail.z;
	if (idx + 3 < size)
		data[idx + 3] = tail.w;
}

""")


class PhiloxGenerator:
	def __init__(self, context, queue, key=None, counter=None, seed=None):
		self.context = context
		self.queue = queue

		self.rounds = 10

		rng = Random(seed)
		iinfo = np.iinfo(np.int32)

		if key is None:
			key = [rng.randrange(int(iinfo.min), int(iinfo.max) + 1) for _ in range(1)]

		if counter is None:
			counter = [rng.randrange(int(iinfo.min), int(iinfo.max) + 1) for _ in range(4)]

		self.key = key

		self.counter = counter
		self.counterMax = iinfo.max

		self.programs = {}
		self.kernels = {}


	def get_kernel(self, transform):
		kernel = self.kernels.get(transform, None)

		if kernel is None:
			tr = "boxMuller" if transform == "normal" else ""
			program = Driver.Program(self.context, randomTmpl.substitute(transform=tr, rounds=self.rounds)).build()

			self.programs[transform] = program
			kernel = program.get_kernel("philox4x32")

			self.kernels[transform] = kernel

		return kernel


	def fill(self, ary, a, b, transform):
		assert ary.dtype == np.float32
		kernel = self.get_kernel(transform)

		n = ary.size // 4

		gridsize, blocks = Driver.splay(self.context, n)
		block = (blocks, 1, 1)
		grid = (gridsize, 1, 1)

		kernel(self.queue, grid, block, ary, ary.size, *self.key, *self.counter, b, a)

		self.counter[0] += ary.size
		incr, self.counter[0] = divmod(self.counter[0], self.counterMax)
		if incr:
			self.counter[1] += incr

			incr, self.counter[1] = divmod(self.counter[1], self.counterMax)
			self.counter[2] += incr


	def fillUniform(self, ary, a=0.0, b=1.0):
		self.fill(ary, a, b, "uniform")


	def fillNormal(self, ary, a=0.0, b=1.0):
		self.fill(ary, a, b, "normal")


def unittest():
	context, queue = create_some_context()
	rng = PhiloxGenerator(context, queue, key=[0], counter=[0, 1, 2, 3])

	data = Driver.empty(queue, shape=(100, ), dtype=np.float32)

	rng.fillUniform(data)
	rng.fillNormal(data, a=1.0, b=2.0)


if __name__ == "__main__":
	unittest()
