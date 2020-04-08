import os, random
from string import Template

from PuzzleLib.Compiler.Toolchain import createTemplateNames, writeTemplates, buildTemplateTest


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


def unittest():
	IntMap = buildTemplateTest(
		name="IntMap", bindingName="TMapTest.c", path="../../TestData", generator=generateMap, K="int", V="int",
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


if __name__ == "__main__":
	unittest()
