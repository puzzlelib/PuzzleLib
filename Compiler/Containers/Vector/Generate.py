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


def unittest():
	IntVector = buildTemplateTest(
		name="IntVector", bindingName="TVectorTest.c", path="../../TestData", generator=generateVector, T="int"
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


if __name__ == "__main__":
	unittest()
