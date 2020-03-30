import os
from string import Template

from PuzzleLib.Compiler.Containers.Tree.Generate import generateTree
from PuzzleLib.Compiler.Toolchain import createTemplateNames, writeTemplates, buildTemplateTest


def generateMalloc(name=None, filename=None):
	treename = generateTree(
		name="AllocTree", K="VoidPtr", V="Allocation",
		headerPreambule=
"""
typedef void *VoidPtr;


typedef struct Allocation
{
	size_t size;
	const char *file;
	int line;
}
Allocation;
""",
		filename=os.path.join(os.path.dirname(filename), "AllocTree")
	)

	name = "TraceMalloc" if name is None else name

	filename = name if filename is None else filename
	headername, bodyname = createTemplateNames(filename)

	dirname = os.path.dirname(__file__)

	with open(os.path.join(dirname, "TMalloc.h"), mode="r", encoding="utf-8") as f:
		header = Template(f.read()).substitute(NAME=name)

	with open(os.path.join(dirname, "TMalloc.c"), mode="r", encoding="utf-8") as f:
		body = Template(f.read()).substitute(HEADER_NAME=os.path.basename(headername), NAME=name)

	writeTemplates([
		(header, headername),
		(body, bodyname)
	])

	return [bodyname, treename]


def unittest():
	TraceMalloc = buildTemplateTest(
		name="TraceMalloc", bindingName="TMallocTest.c", path="../TestData", generator=generateMalloc,
		defines=["ENABLE_TRACE_MALLOC"]
	)

	ptr = TraceMalloc.malloc(16)

	leaks = TraceMalloc.traceLeaks()
	assert len(leaks) == 1

	TraceMalloc.free(ptr)

	leaks = TraceMalloc.traceLeaks()
	assert len(leaks) == 0


if __name__ == "__main__":
	unittest()
