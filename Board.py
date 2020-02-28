from graphviz import Digraph


def drawBoard(net, filename, view=True, fmt="svg", modulesOnly=False, name=None, fontname="Consolas", fullnames=True):
	if name is None:
		name = net.name

	g = Digraph(name, filename=filename)
	g.format = fmt

	g.attr(label=name, labelloc="top", labeljust="center", fontcolor="#31343F", fontname=fontname)
	g.edge_attr.update(color="#31343F")
	g.node_attr.update(style="filled", color="#CA5237", shape="Mrecord", fontname=fontname, fontcolor="white",
					   fontsize="8")

	blueprint = net.getBlueprint()
	drawGraph(g, blueprint, childName=name, modulesOnly=modulesOnly, fullnames=fullnames)

	g.view(filename) if view else g.render(filename)


def buildContainerLabel(classname, params, name, showFullname):
	label = """<
	<table border="0" cellspacing="5" bgcolor="#FFB84D" style="rounded">
		<tr><td align="center" colspan="2"><font point-size="10">%s</font></td></tr>
	""" % classname

	params = params.copy()

	if showFullname:
		params["fullname"] = name

	for paramName in sorted(params.keys()):
		label += "<tr><td align=\"left\">%s</td><td align=\"right\">%s</td></tr>" % (paramName, params[paramName])

	label += "</table>>"
	return label


def buildModuleLabel(classname, params, name, showFullname):
	label = """<
	<table cellspacing="0">
		<tr><td align="center" colspan="2"><font point-size="10">%s</font></td></tr>
	""" % classname

	params = params.copy()

	if showFullname:
		params["fullname"] = name

	for paramName in sorted(params.keys()):
		color = "white"
		if paramName == "name":
			color = "#31343F"

		label += "<tr><td align=\"left\"><font color=\"%s\">%s</font></td>" \
				 "<td align=\"right\"><font color=\"%s\">%s</font></td></tr>" % \
				 (color, paramName, color, params[paramName])

	label += "</table>>"
	return label


def drawGraph(g, blueprint, parentName=None, childName=None, clusterIdx=0, modulesOnly=False, fullnames=True):
	classname = blueprint["classname"]
	scheme = blueprint["scheme"]

	name = "%s.%s" % (parentName, childName) if parentName is not None else str(childName)

	if classname in {"Sequential", "Parallel", "Graph"}:
		graph = blueprint["graph"]
		elements = blueprint["modules"]

		with g.subgraph(name="cluster_%s" % clusterIdx) as c:
			clusterIdx += 1

			if not modulesOnly:
				c.attr(label=buildContainerLabel(classname, {"name": scheme["name"]}, name, fullnames),
					   labeljust="right", shape="Mrecord", color="#31343F",
					   fontcolor="#554037", fontsize="8", rankdir="TB")
			else:
				c.attr(color="#FFFFFF", fontcolor="#FFFFFF")

			inNodes, outNodes = [], []
			if classname == "Sequential":
				if len(graph) > 0:
					clusterIdx, inNodes, outNodes = drawGraph(c, elements[graph[0]], parentName=name,
															  childName=graph[0], clusterIdx=clusterIdx,
															  modulesOnly=modulesOnly, fullnames=fullnames)

				curOutNodes = outNodes
				for i, nm in enumerate(graph[1:]):
					clusterIdx, newInNodes, outNodes = drawGraph(c, elements[nm], parentName=name,
																 childName=nm, clusterIdx=clusterIdx,
																 modulesOnly=modulesOnly, fullnames=fullnames)

					connectNodes(c, curOutNodes, newInNodes)
					curOutNodes = outNodes

				return clusterIdx, [inNode + ":w" for inNode in inNodes if isinstance(inNode, str)], outNodes

			elif classname == "Parallel":
				for nm in graph:
					clusterIdx, newInNodes, newOutNodes = drawGraph(c, elements[nm], parentName=name,
																	childName=nm, clusterIdx=clusterIdx,
																	modulesOnly=modulesOnly, fullnames=fullnames)

					inNodes.append(newInNodes)
					outNodes.append(newOutNodes)

				return clusterIdx, inNodes, outNodes

			elif classname == "Graph":
				inputs, outputs = set(blueprint["inputs"]), set(blueprint["outputs"])
				nodes = {}

				for nm, mod in elements.items():
					_, newInNodes, newOutNodes = drawGraph(c, mod, parentName=name, childName=nm, clusterIdx=clusterIdx,
														   modulesOnly=modulesOnly, fullnames=fullnames)

					nodes[nm] = (newInNodes, newOutNodes)

					if nm in inputs:
						inNodes.extend(newInNodes)

					if nm in outputs:
						outNodes.extend(newOutNodes)

				for nm, node in nodes.items():
					connectNodes(c, [nodes[name][0] for name, _ in graph[nm]], node[1])

				return clusterIdx, inNodes, outNodes

			else:
				raise NotImplementedError(classname)

	else:
		g.node(name, label=buildModuleLabel(classname, scheme, name, fullnames))
		return clusterIdx, [name], [name]


def connectNodes(g, inNodes, outNodes):
	if isinstance(inNodes, str):
		if isinstance(outNodes, str):
			g.edges([(inNodes, outNodes)])
		else:
			for outNode in outNodes:
				connectNodes(g, inNodes, outNode)

	elif isinstance(outNodes, str):
		for inNode in inNodes:
			connectNodes(g, inNode, outNodes)

	elif len(inNodes) == len(outNodes):
		for j, node in enumerate(outNodes):
			connectNodes(g, inNodes[j], node)

	elif len(inNodes) == 1:
		for node in outNodes:
			connectNodes(g, inNodes[0], node)

	elif len(outNodes) == 1:
		for node in inNodes:
			connectNodes(g, node, outNodes[0])

	else:
		assert False


def netTest():
	from PuzzleLib.Models.Nets.Inception import loadInceptionV3
	net = loadInceptionV3(None)

	drawBoard(net, filename="./TestData/net.gv", view=False, modulesOnly=False)


def graphTest():
	from PuzzleLib.Models.Nets.ResNet import loadResNet
	net = loadResNet(None, layers="50")

	from PuzzleLib.Passes.ConvertToGraph import toGraph
	graph = toGraph(net, nodesOnly=True)

	drawBoard(graph, filename="./TestData/graph.gv", view=False, modulesOnly=False)


def mixedTest():
	from PuzzleLib.Containers import Graph, Sequential
	from PuzzleLib.Modules import Linear, Split, Concat, Activation, relu

	v1 = Linear(100, 50, name="v1").node()
	h1 = Split(axis=1, sections=(20, 20, 10), name="h1").node(v1)

	v2 = Linear(100, 50, name="v2").node()
	h2 = Concat(axis=1, name="h2").node((h1, [1, 2]), v2)
	h3 = Activation(relu, name="h3").node(h2)

	h4 = Concat(axis=1, name="h4").node((h1, 0), h3)
	mlp = Graph(inputs=[v1, v2], outputs=h4)

	seq = Sequential()

	seq.append(Linear(10, 200))
	seq.append(Split(axis=1, sections=(100, 100)))

	seq.append(mlp)
	seq.append(Activation(relu))

	drawBoard(seq, filename="./TestData/mixed.gv", view=False, modulesOnly=False)


def unittest():
	netTest()
	graphTest()
	mixedTest()


if __name__ == "__main__":
	unittest()
