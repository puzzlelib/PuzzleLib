from string import Template
import numpy as np


scanSumTmpl = Template("""

typedef struct ScanResult
{
	unsigned scan, reduction;
}
ScanResult;


typedef struct ScanStorage
{
	unsigned sums[$warpSize];
}
ScanStorage;


__forceinline__ __device__ unsigned warpScanSum(unsigned value)
{
	int laneId = threadIdx.x % $warpSize;

	for (int i = 1; i <= $warpSize; i *= 2)
	{
		unsigned downval = __shfl_up_sync((unsigned)-1, value, i, $warpSize);
		if (laneId >= i) value += downval;
	}

	return value;
}


__forceinline__ __device__ ScanResult blockScanSum(unsigned value, ScanStorage *storage)
{
	unsigned initval = value;
	int warpId = threadIdx.x / $warpSize, laneId = threadIdx.x % $warpSize;

	value = warpScanSum(value);

	if (laneId == $warpSize - 1)
		storage->sums[warpId] = value;

	__syncthreads();

	if (warpId == 0)
		storage->sums[laneId] = warpScanSum((laneId < $NT / $warpSize) ? storage->sums[laneId] : 0);

	__syncthreads();

	unsigned blockSum = 0;
	if (warpId > 0)
		blockSum = storage->sums[warpId - 1];

	unsigned reduction = storage->sums[$NT / $warpSize - 1];
	__syncthreads();

	ScanResult result = {blockSum + value - initval, reduction};
	return result;
}

""")


radixSortTmpl = Template("""

$scanSum


typedef unsigned short DigitCounter;
typedef unsigned PackCounter;


enum
{
	RadixSort_Bits = 6,
	RadixSort_Lanes = 1 << (RadixSort_Bits - 1),
	RadixSort_PackRatio = sizeof(PackCounter) / sizeof(DigitCounter)
};


typedef struct SortStorage
{
	union
	{
		DigitCounter bins[(RadixSort_Lanes + 1) * $NT * RadixSort_PackRatio];
		PackCounter grid[$NT * (RadixSort_Lanes + 1)];
	}
	hist;

	ScanStorage scanStorage;
}
SortStorage;


__forceinline__ __device__
void blockRadixSort(int keys[$VT], int values[$VT], int outkeys[$NV], int outvalues[$NV], SortStorage *storage)
{
	int currBit = 0;

	while (true)
	{
		for (int i = threadIdx.x; i < $NT * (RadixSort_Lanes + 1); i += $NT)
			storage->hist.grid[i] = 0;

		__syncthreads();
		DigitCounter binOffsets[$VT], *binPtrs[$VT];

		for (int i = 0; i < $VT; i++)
		{
			int radix = (keys[i] >> currBit) & ((1 << RadixSort_Bits) - 1);
			int subcounter = radix >> (RadixSort_Bits - 1), digit = radix & (RadixSort_Lanes - 1);

			binPtrs[i] = storage->hist.bins + (digit * $NT + threadIdx.x) * RadixSort_PackRatio + subcounter;
			binOffsets[i] = *binPtrs[i];

			*binPtrs[i] += 1;
		}
		__syncthreads();

		PackCounter laneCache[RadixSort_Lanes + 1];
		PackCounter upsweep = 0;

		for (int i = 0; i < RadixSort_Lanes + 1; i++)
		{
			laneCache[i] = storage->hist.grid[threadIdx.x * (RadixSort_Lanes + 1) + i];
			upsweep += laneCache[i];
		}

		ScanResult result = blockScanSum(upsweep, &storage->scanStorage);
		PackCounter downsweep = result.scan + (result.reduction << (sizeof(DigitCounter) * 8));

		for (int i = 0; i < RadixSort_Lanes + 1; i++)
		{
			storage->hist.grid[threadIdx.x * (RadixSort_Lanes + 1) + i] = downsweep;
			downsweep += laneCache[i];
		}
		__syncthreads();

		for (int i = 0; i < $VT; i++)
		{
			int rank = *binPtrs[i] + binOffsets[i];
			outkeys[rank] = keys[i], outvalues[rank] = values[i];
		}
		__syncthreads();

		currBit += RadixSort_Bits;
		if (currBit >= sizeof(keys[0]) * 8)
			break;

		for (int i = 0; i < $VT; i++)
		{
			int index = threadIdx.x * $VT + i;
			keys[i] = outkeys[index], values[i] = outvalues[index];
		}
	}
}

""")


segmentSeqTmpl = Template("""

$radixSort


typedef struct SegmentResult
{
	int start[$VT], end[$VT], label[$VT];
	int length;
}
SegmentResult;


typedef struct SegmentStorage
{
	union
	{
		SortStorage sortStorage;
		ScanStorage scanStorage;
		int labels[$NV];
	};

	int keys[$NV], offsets[$NV];
}
SegmentStorage;


__forceinline__ __device__
SegmentResult blockSegmentSeq(int keys[$VT], int length, int indices[$NV], SegmentStorage *storage)
{
	int values[$VT];

	for (int i = 0; i < $VT; i++)
		values[i] = threadIdx.x + i * $NT;

	blockRadixSort(keys, values, storage->keys, indices, &storage->sortStorage);

	unsigned splitFlags = 0;
	int key = storage->keys[threadIdx.x * $VT];

	for (int i = 0; i < $VT; i++)
	{
		int index = threadIdx.x * $VT + i + 1;
		if (index < length)
		{
			int next = storage->keys[index];

			if (key != next)
				splitFlags |= 1 << i;

			key = next;
		}
		else
		{
			if (index == length)
				splitFlags |= 1 << i;

			break;
		}
	}

	ScanResult scanResult = blockScanSum(__popc(splitFlags), &storage->scanStorage);
	unsigned offset = scanResult.scan, uniqueKeys = scanResult.reduction;

	for (int i = 0; i < $VT; i++)
	{
		if (splitFlags & (1 << i))
		{
			int index = threadIdx.x * $VT + i;

			storage->offsets[offset] = index + 1;
			storage->labels[offset] = storage->keys[index];

			offset += 1;
		}
	}
	__syncthreads();

	SegmentResult result;
	result.length = 0;

	for (int i = 0; i < $VT; i++)
	{
		int index = threadIdx.x + i * $NT;
		if (index < uniqueKeys)
		{
			result.start[i] = (index > 0) ? storage->offsets[index - 1] : 0;
			result.end[i] = storage->offsets[index];

			result.label[i] = storage->labels[index];
			result.length += 1;
		}
	}
	 __syncthreads();

	return result;
}

""")


scanSumTestTmpl = Template("""

$scanSum


extern "C"
__global__ void scanSum(unsigned *outdata, const unsigned *indata, int length)
{
	__shared__ ScanStorage scanStorage;

	unsigned key = (threadIdx.x < length) ? indata[threadIdx.x] : (unsigned)-1;
	ScanResult result = blockScanSum(key, &scanStorage);

	if (threadIdx.x < length)
		outdata[threadIdx.x] = result.scan;
}

""")


radixSortTestTmpl = Template("""

$radixSort


extern "C"
__global__ void radixSort(int *outkeys, int *outvalues, const int *inkeys, const int *invalues, int length)
{
	__shared__ int shkeys[$NV], shvalues[$NV];
	__shared__ SortStorage storage;

	int keys[$VT], values[$VT];

	for (int i = 0; i < $VT; i++)
	{
		int j = threadIdx.x + i * $NT;

		keys[i] = (j < length) ? inkeys[j] : 0x7FFFFFFF;
		values[i] = (j < length) ? invalues[j] : 0;
	}
 
	blockRadixSort(keys, values, shkeys, shvalues, &storage);

	for (int i = threadIdx.x; i < length; i += $NT)
		outkeys[i] = shkeys[i], outvalues[i] = shvalues[i];
}

""")


segmentTestTmpl = Template("""

$segmentSeq


extern "C"
__global__ void segmentSeq(int *segments, int *indices, const int *data, int length)
{
	__shared__ int shIndices[$NV];
	__shared__ SegmentStorage storage;

	int keys[$VT];

	for (int i = 0; i < $VT; i++)
	{
		int j = threadIdx.x + i * $NT;
		keys[i] = (j < length) ? data[j] : 0x7FFFFFFF;
	}

	SegmentResult result = blockSegmentSeq(keys, length, shIndices, &storage);
	int3 *segments3 = (int3 *)segments;

	for (int i = 0; i < $VT; i++)
	{
		int j = threadIdx.x + i * $NT;

		if (j < length)
		{
			segments3[j] = (i < result.length) ?
				make_int3(result.start[i], result.end[i], result.label[i]) : make_int3(-1, -1, -1);

			indices[j] = shIndices[j];
		}
	}
}

""")


class RadixSortModule:
	def __init__(self, backend):
		self.GPUArray = backend.GPUArray

		self.NT, self.VT = 128, 2
		self.NV = self.NT * self.VT

		scanSum = scanSumTmpl.substitute(warpSize=backend.warpSize, NT=self.NT)
		self.scanMod = backend.SourceModule(scanSumTestTmpl.substitute(scanSum=scanSum))

		radixSort = radixSortTmpl.substitute(
			scanSum=scanSum, warpSize=backend.warpSize, NT=self.NT, VT=self.VT, NV=self.NV
		)
		self.radixMod = backend.SourceModule(radixSortTestTmpl.substitute(
			radixSort=radixSort, NT=self.NT, VT=self.VT, NV=self.NV
		))

		segmentSeq = segmentSeqTmpl.substitute(radixSort=radixSort, NT=self.NT, VT=self.VT, NV=self.NV)
		self.segmentMod = backend.SourceModule(segmentTestTmpl.substitute(
			segmentSeq=segmentSeq, NT=self.NT, VT=self.VT, NV=self.NV
		))


	def scanSum(self, data):
		assert data.dtype == np.uint32

		length, = data.shape
		assert length <= self.NT

		outdata = self.GPUArray.empty(data.shape, dtype=data.dtype)

		self.scanMod.scanSum(outdata, data, np.int32(length), block=(self.NT, 1, 1), grid=(1, 1, 1))
		return outdata


	def radixSort(self, keys, values):
		assert keys.dtype == np.int32 and values.dtype == np.int32
		assert keys.shape == values.shape

		length, = keys.shape
		assert length <= self.NV

		outkeys = self.GPUArray.empty(keys.shape, dtype=keys.dtype)
		outvalues = self.GPUArray.empty(values.shape, dtype=values.dtype)

		self.radixMod.radixSort(
			outkeys, outvalues, keys, values, np.int32(length), block=(self.NT, 1, 1), grid=(1, 1, 1)
		)
		return outkeys, outvalues


	def segmentSeq(self, data):
		assert data.dtype == np.int32

		length, = data.shape
		assert length <= self.NV

		segments = self.GPUArray.empty((length, 3), dtype=np.int32)
		indices = self.GPUArray.empty(data.shape, dtype=np.int32)

		self.segmentMod.segmentSeq(segments, indices, data, np.int32(length), block=(self.NT, 1, 1), grid=(1, 1, 1))
		return segments, indices


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = RadixSortModule(Backend.getBackend(deviceIdx))

		scanSumTest(module)
		radixSortTest(module)
		segmentTest(module)


def scanSumTest(module):
	hostData = np.random.randint(0, 1000, size=(120, ), dtype=np.uint32)
	outdata = module.scanSum(module.GPUArray.toGpu(hostData))

	hostOutData = np.empty_like(hostData)

	hostOutData[0] = 0
	hostOutData[1:] = np.cumsum(hostData)[:-1]

	assert np.allclose(outdata.get(), hostOutData)


def radixSortTest(module):
	hostKeys = np.random.randint(0, (1 << 31) - 1, size=(250, ), dtype=np.int32)
	hostValues = np.arange(0, hostKeys.shape[0], dtype=np.int32)

	outkeys, outvalues = module.radixSort(module.GPUArray.toGpu(hostKeys), module.GPUArray.toGpu(hostValues))

	assert (outkeys.get() == np.sort(hostKeys)).all()
	assert (outvalues.get() == np.argsort(hostKeys)).all()


def segmentTest(module):
	hostData = np.random.randint(10, 30, size=(250, ), dtype=np.int32)
	segments, indices = module.segmentSeq(module.GPUArray.toGpu(hostData))

	hostSortedData = np.sort(hostData)
	hostSegments = np.empty(shape=segments.shape, dtype=segments.dtype)

	segIndex = 0
	hostSegments[segIndex, 0] = 0

	for i in range(hostData.shape[0] - 1):
		if hostSortedData[i] != hostSortedData[i + 1]:
			hostSegments[segIndex, 1] = hostSegments[segIndex + 1, 0] = i + 1
			hostSegments[segIndex, 2] = hostSortedData[i]

			segIndex += 1

	hostSegments[segIndex, 1] = hostData.shape[0]
	hostSegments[segIndex, 2] = hostSortedData[-1]

	hostSegments[segIndex + 1:] = -1

	assert (segments.get() == hostSegments).all()
	assert (hostData[indices.get()] == np.sort(hostData)).all()


if __name__ == "__main__":
	unittest()
