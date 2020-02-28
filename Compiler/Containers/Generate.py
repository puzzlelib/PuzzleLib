import os, random
from string import Template

from PuzzleLib.Compiler.Toolchain import createTemplateNames, writeTemplates, buildTemplateTest


def generateVector(name, T, destruct="(void)", minCapacity=16, headerPreambule=None, bodyPreambule=None,
				   malloc="malloc", free="free", filename=None):
	headerPreambule = "%s\n\n" % headerPreambule if headerPreambule is not None else ""
	bodyPreambule = "%s\n\n" % bodyPreambule if bodyPreambule is not None else ""

	filename = name if filename is None else filename
	headername, bodyname = createTemplateNames(filename)

	dirname = os.path.dirname(__file__)
	headerTmpl, bodyTmpl = os.path.join(dirname, "TVector.h"), os.path.join(dirname, "TVector.c")

	with open(headerTmpl, mode="r", encoding="utf-8") as f:
		header = Template(f.read()).substitute(HEADER_PREAMBULE=headerPreambule, NAME=name, T=T)

	with open(bodyTmpl, mode="r", encoding="utf-8") as f:
		body = Template(f.read()).substitute(
			HEADER_NAME=os.path.basename(headername), BODY_PREAMBULE=bodyPreambule, NAME=name, T=T,
			MIN_CAPACITY=minCapacity, MALLOC=malloc, FREE=free, DESTRUCT=destruct
		)

	writeTemplates([
		(header, headername),
		(body, bodyname)
	])

	return bodyname


def generateMap(name, K, V, hasher, compareKeys, borrowKey, borrowValue,
				destructKey="(void)", destructValue="(void)", minLog2Capacity=4,
				headerPreambule=None, bodyPreambule=None, malloc="malloc", free="free", filename=None):
	headerPreambule = "%s\n\n" % headerPreambule if headerPreambule is not None else ""
	bodyPreambule = "%s\n\n" % bodyPreambule if bodyPreambule is not None else ""

	filename = name if filename is None else filename
	headername, bodyname = createTemplateNames(filename)

	dirname = os.path.dirname(__file__)
	headerTmpl, bodyTmpl = os.path.join(dirname, "TMap.h"), os.path.join(dirname, "TMap.c")

	with open(headerTmpl, mode="r", encoding="utf-8") as f:
		header = Template(f.read()).substitute(HEADER_PREAMBULE=headerPreambule, NAME=name, K=K, V=V)

	with open(bodyTmpl, mode="r", encoding="utf-8") as f:
		body = Template(f.read()).substitute(
			HEADER_NAME=os.path.basename(headername), BODY_PREAMBULE=bodyPreambule, NAME=name, K=K, V=V,
			MIN_LOG2_CAPACITY=minLog2Capacity, MALLOC=malloc, FREE=free,
			HASHER=hasher, COMPARE_KEYS=compareKeys, BORROW_KEY=borrowKey, BORROW_VALUE=borrowValue,
			DESTRUCT_KEY=destructKey, DESTRUCT_VALUE=destructValue
		)

	writeTemplates([
		(header, headername),
		(body, bodyname)
	])

	return bodyname


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


def vectorTest():
	IntVector = buildTemplateTest(
		name="IntVector", bindingName="TVectorTest.c", path="../TestData", generator=generateVector, T="int"
	)

	size = 1 << 16

	pyvec = list(range(size))
	random.shuffle(pyvec)

	vector = IntVector.IntVector()

	for i in pyvec:
		vector.append(i)

	assert len(vector) == size

	for i in range(size):
		assert vector[i] == pyvec[i]

	for i in reversed(pyvec):
		assert vector.pop() == i

	assert len(vector) == 0


def mapTest():
	IntMap = buildTemplateTest(
		name="IntMap", bindingName="TMapTest.c", path="../TestData", generator=generateMap, K="int", V="int",
		hasher="hashKey", compareKeys="compareKeys", borrowKey="(int)", borrowValue="(int)",
		bodyPreambule="""
inline static size_t hashKey(int key) { return key; }
inline static bool compareKeys(int key1, int key2) { return key1 == key2; }
""")

	size = 1 << 16

	keys, values = list(range(size)), list(range(size))
	random.shuffle(keys)
	random.shuffle(values)

	pymap = {k: v for k, v in zip(keys, values)}

	intmap = IntMap.IntMap()

	for k, v in pymap.items():
		intmap[k] = v

	assert len(intmap) == size

	for k in pymap.keys():
		assert intmap[k] == pymap[k]

	for k in pymap.keys():
		del intmap[k]

	assert len(intmap) == 0


def treeTest():
	IntTree = buildTemplateTest(
		name="IntTree", bindingName="TTreeTest.c", path="../TestData", generator=generateTree, K="int", V="int"
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


def unittest():
	vectorTest()
	mapTest()
	treeTest()


if __name__ == "__main__":
	unittest()
