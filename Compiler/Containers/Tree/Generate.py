import os, random
from string import Template

from PuzzleLib.Compiler.Toolchain import createTemplateNames, writeTemplates, buildTemplateTest


def generateTree(name, K, V, headerPreambule=None, bodyPreambule=None, malloc="malloc", free="free", filename=None):
	headerPreambule = "%s\n\n" % headerPreambule if headerPreambule is not None else ""
	bodyPreambule = "%s\n\n" % bodyPreambule if bodyPreambule is not None else ""

	filename = name if filename is None else filename
	headername, bodyname = createTemplateNames(filename)

	dirname = os.path.dirname(__file__)
	headerTmpl, bodyTmpl = os.path.join(dirname, "TTree.h"), os.path.join(dirname, "TTree.c")

	with open(headerTmpl, mode="r", encoding="utf-8") as f:
		header = Template(f.read()).substitute(HEADER_PREAMBULE=headerPreambule, NAME=name, K=K, V=V)

	with open(bodyTmpl, mode="r", encoding="utf-8") as f:
		body = Template(f.read()).substitute(
			HEADER_NAME=os.path.basename(headername), BODY_PREAMBULE=bodyPreambule, NAME=name, K=K, V=V,
			MALLOC=malloc, FREE=free
		)

	writeTemplates([
		(header, headername),
		(body, bodyname)
	])

	return bodyname


def unittest():
	IntTree = buildTemplateTest(
		name="IntTree", bindingName="TTreeTest.c", path="../../TestData", generator=generateTree, K="int", V="int"
	)

	size = 1 << 16

	keys, values = list(range(size)), list(range(size))
	random.shuffle(keys)
	random.shuffle(values)

	pytree = {k: v for k, v in zip(keys, values)}

	inttree = IntTree.IntTree()

	for k, v in pytree.items():
		inttree[k] = v

	assert len(inttree) == size
	assert inttree.validate()

	for k in pytree.keys():
		assert inttree[k] == pytree[k]

	for k in pytree.keys():
		del inttree[k]

	assert len(inttree) == 0
	assert inttree.validate()


if __name__ == "__main__":
	unittest()
