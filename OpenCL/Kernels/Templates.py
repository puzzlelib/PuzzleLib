import re
from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver
from PuzzleLib.OpenCL.Kernels.Utils import nthreads, roundUpDiv


def create_some_context():
	platforms = Driver.get_platforms()

	for vendor in ["AMD Accelerated Parallel Processing"]:
		platform = next((pl for pl in platforms if pl.name == vendor), None)
		if platform is not None:
			break

	assert platform is not None
	device = platform.get_devices(type=Driver.device_type.GPU)[0]

	context = Driver.Context([device])
	queue = Driver.CommandQueue(context, profiling=True)

	return context, queue


eltwiseTmpl = Template("""

$preamble


__kernel void $name($arguments, long n)
{
	int index = get_local_id(0);
	int gsize = get_global_size(0);
	int groupStart = get_local_size(0) * get_group_id(0);

	$argPrep

	for (long i = groupStart + index; i < n; i += gsize)
	{
		$operation;
	}
}


__kernel void ${name}_strided($arguments, long start, long step, long stop)
{
	int index = get_local_id(0);
	int gsize = get_global_size(0);
	int groupStart = get_local_size(0) * get_group_id(0);

	$argPrep

	for (long i = start + (groupStart + index) * step; i < stop; i += gsize * step)
	{
		$operation;
	}
}

""")


reduceTmpl = Template("""

#undef GROUP_SIZE
#undef READ_AND_MAP
#undef REDUCE
#undef OUT_TYPE


#define GROUP_SIZE $groupSize
#define READ_AND_MAP(i) ($mapExpr)
#define REDUCE(a, b) ($reduceExpr)
#define OUT_TYPE $outtype


__kernel void $name(__global OUT_TYPE *out, $arguments, unsigned seqCount, long n)
{
	__local OUT_TYPE ldata[GROUP_SIZE];
	unsigned lid = get_local_id(0);

	long baseIdx = get_group_id(0) * GROUP_SIZE * seqCount + lid;

	$argPrep

	OUT_TYPE acc = $neutral;
	for (unsigned i = 0; i < seqCount; i++)
	{
		if (baseIdx >= n)
			break;

		acc = REDUCE(acc, READ_AND_MAP(baseIdx));
		baseIdx += GROUP_SIZE;
	}

	ldata[lid] = acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	#if GROUP_SIZE >= 256
		if (lid < 128) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 128]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 128
		if (lid < 64) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 64]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 64
		if (lid < 32) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 32]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 32
		if (lid < 16) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 16]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 16
		if (lid < 8) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 8]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 8
		if (lid < 4) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 4]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 4
		if (lid < 2) { ldata[lid] = REDUCE(ldata[lid], ldata[lid + 2]); barrier(CLK_LOCAL_MEM_FENCE); }
	#endif

	#if GROUP_SIZE >= 2
		ldata[lid] = REDUCE(ldata[lid], ldata[lid + 1]);
	#endif

	if (lid == 0) out[get_group_id(0)] = ldata[0];
}

""")


def formatArgs(arguments):
	arguments = ", ".join("%s%s" % ("__global " if "*" in arg else "", arg) for arg in arguments.split(sep=","))
	return arguments


def createAdjustments(arguments, expr):
	lastWordRe = re.compile(r"\w+$")
	wordsRe = re.compile(r"\b(\w+)\b")

	outarguments = []
	argPrep = ""

	subs = {}
	for arg in arguments.split(sep=","):
		if "*" not in arg:
			outarguments.append(arg)
			continue

		m = lastWordRe.search(arg)
		identifier = m.group(0)

		subs[identifier] = "%s_s" % identifier
		typ = arg[:m.start()]

		argPrep += "__global %s%s_s = (__global %s)(%s + %s_offset);" % (typ, identifier, typ, identifier, identifier)
		outarguments.extend([arg, "long %s_offset" % identifier])

	expr = wordsRe.sub(lambda w: subs.get(w.group(0), w.group(0)), expr)

	outarguments = formatArgs(", ".join(outarguments))
	return outarguments, argPrep, expr


def rewriteArgs(args):
	size = 0
	outargs = []

	for arg in args:
		if hasattr(arg, "size"):
			size = max(arg.size, size)

		if hasattr(arg, "item_offset"):
			outargs.extend([arg, arg.item_offset])
		else:
			outargs.append(arg)

	return outargs, size


def ctypeFromDtype(dtype):
	if dtype == np.float32:
		ctype = "float"
	elif dtype == np.float64:
		ctype = "double"
	elif dtype == np.int32:
		ctype = "int"
	elif dtype == np.uint32:
		ctype = "unsigned int"
	else:
		raise NotImplementedError()

	return ctype


class ElementwiseKernel:
	def __init__(self, arguments, operation, name, context=None, queue=None, preamble=None):
		self.context = context
		self.queue = queue

		self.arguments = arguments
		self.operation = operation
		self.name = name

		self.preamble = preamble

		self.program = None
		self.krl, self.krlStrided = None, None


	def get_kernel(self, strided):
		if self.program is None:
			arguments, argPrep, operation = createAdjustments(self.arguments, self.operation)

			source = eltwiseTmpl.substitute(name=self.name, arguments=arguments, operation=operation, argPrep=argPrep,
											preamble="" if self.preamble is None else self.preamble)

			self.program = Driver.Program(self.context, source).build()

			self.krl = self.program.get_kernel(self.name)
			self.krlStrided = self.program.get_kernel("%s%s" % (self.name, "_strided"))

		return self.krlStrided if strided else self.krl


	def __call__(self, *args, **kwargs):
		slc = kwargs.get("slice", None)
		kernel = self.get_kernel(slc is not None)

		if slc is not None:
			start, step, stop = slc[:]

			args, size = rewriteArgs(args)
			if stop is None:
				stop = size

			gridsize, blocks = Driver.splay(self.context, (stop - start) // step)
			block = (blocks, 1, 1)
			grid = (gridsize, 1, 1)

			kernel(self.queue, grid, block, *args, start, step, stop)

		else:
			args, size = rewriteArgs(args)

			gridsize, blocks = Driver.splay(self.context, size)
			block = (blocks, 1, 1)
			grid = (gridsize, 1, 1)

			kernel(self.queue, grid, block, *args, size)


class ReductionKernel:
	def __init__(self, context, queue, outtype, neutral, reduce_expr, map_expr, arguments, name="reduce"):
		self.context = context
		self.queue = queue

		self.outtype = outtype
		self.neutral = neutral
		self.reduceExpr = reduce_expr
		self.mapExpr = map_expr
		self.arguments = arguments

		self.name = name
		self.program = None
		self.kernels = None

		self.maxGroupCount = 1024
		self.smallSeqCount = 4


	def get_kernels(self):
		if self.program is None:
			outtype = ctypeFromDtype(self.outtype)

			name1 = "%s_%s" % (self.name, "stage1")
			name2 = "%s_%s" % (self.name, "stage2")

			arguments, argPrep, mapExpr = createAdjustments(self.arguments, self.mapExpr)
			code1 = reduceTmpl.substitute(outtype=outtype, neutral=self.neutral, reduceExpr=self.reduceExpr,
										  argPrep=argPrep, mapExpr=mapExpr, arguments=arguments, groupSize=nthreads,
										  name=name1)

			arguments, argPrep, mapExpr = createAdjustments("OUT_TYPE *data", "data[i]")
			code2 = reduceTmpl.substitute(outtype=outtype, neutral=self.neutral, reduceExpr=self.reduceExpr,
										  argPrep=argPrep, mapExpr=mapExpr, arguments=arguments, groupSize=nthreads,
										  name=name2)

			self.program = Driver.Program(self.context, code1 + code2).build()
			self.kernels = (self.program.get_kernel(name1), self.program.get_kernel(name2))

		return self.kernels


	def __call__(self, *args, **kwargs):
		allocator = kwargs.get("allocator", None)

		stage1, stage2 = self.get_kernels()
		kernel = stage1

		while True:
			args, size = rewriteArgs(args)

			if size <= nthreads * self.smallSeqCount * self.maxGroupCount:
				groupCount = roundUpDiv(size, self.smallSeqCount * nthreads)
				seqCount = self.smallSeqCount
			else:
				groupCount = self.maxGroupCount
				seqCount = roundUpDiv(size, groupCount * nthreads)

			if groupCount == 1:
				result = Driver.empty(self.queue, (), dtype=self.outtype, allocator=allocator)
			else:
				result = Driver.empty(self.queue, (groupCount, ), dtype=self.outtype, allocator=allocator)

			kernel(self.queue, (groupCount * nthreads, 1, 1), (nthreads, 1, 1), result, *args, seqCount, size)

			if groupCount == 1:
				break
			else:
				kernel = stage2
				args = [result, ]

		return result


minreductions = {}
maxreductions = {}


def minmax(context, queue, ary, typ):
	dtype = ary.dtype
	info = np.finfo if issubclass(dtype.type, np.floating) else np.iinfo

	if typ == "min":
		reductions = minreductions

		expr = "a < b ? a : b"
		ctype = ctypeFromDtype(ary.dtype)
		neutral = info(dtype).max

	elif typ == "max":
		reductions = maxreductions
		expr = "a < b ? b : a"
		ctype = ctypeFromDtype(ary.dtype)
		neutral = info(dtype).min

	else:
		raise ValueError()

	reduction = reductions.get((context, ary.dtype), None)

	if reduction is None:
		reduction = ReductionKernel(context, None, ary.dtype, neutral=str(neutral),
									reduce_expr=expr, map_expr="data[i]", arguments="const %s *data" % ctype)

		reductions[(context, queue)] = reduction

	reduction.queue = queue
	return reduction(ary)


def minimum(context, queue, ary):
	return minmax(context, queue, ary, "min")


def maximum(context, queue, ary):
	return minmax(context, queue, ary, "max")


def unittest():
	context, queue = create_some_context()

	eltwiseTest(context, queue)
	reduceTest(context, queue)
	minmaxTest(context, queue)


def eltwiseTest(context, queue):
	outdata = Driver.empty(queue, (1 << 18, ), dtype=np.int32)
	indata = Driver.to_device(queue, np.random.randint(0, 1000, size=(1 << 18, ), dtype=np.int32))

	krl = ElementwiseKernel(
		"int *outdata, const int *indata",
		"outdata[i] = indata[i] * indata[i]",
		"krl",
		context, queue
	)

	krl(outdata, indata)

	hostOutData = indata.get() * indata.get()
	assert np.allclose(hostOutData, outdata.get())


def reduceTest(context, queue):
	data = Driver.to_device(queue, np.random.randn(1 << 18).astype(np.float32))

	krl = ReductionKernel(context, queue, np.float32, neutral="0.0f",
						  reduce_expr="a + b", map_expr="data[i]", arguments="const float *data")

	sm = krl(data)
	hostSum = np.sum(data.get())

	assert np.isclose(hostSum, sm.get())


def minmaxTest(context, queue):
	data = Driver.to_device(queue, np.random.randn(1 << 18).astype(np.float32))

	mn, mx = minimum(context, queue, data), maximum(context, queue, data)

	hostData = data.get()
	hostMin, hostMax = np.min(hostData), np.max(hostData)

	assert np.isclose(hostMin, mn.get())
	assert np.isclose(hostMax, mx.get())


if __name__ == "__main__":
	unittest()
