import os, json, itertools

from PuzzleLib.Compiler.Compilers.Compiler import CompilerError
from PuzzleLib.Compiler.Toolchain import guessToolchain, loadDynamicModule


class BuildError(Exception):
	pass


class Rule:
	def __init__(self, deps, toolchain=None, target=None, ext=None):
		self.deps, self.toolchain = deps, toolchain
		self.target = target if target is not None else self.inferTarget(deps, toolchain.oext if ext is None else ext)


	@staticmethod
	def inferTarget(deps, ext):
		return "%s%s" % (os.path.splitext(deps[0])[0], ext)


	def toString(self):
		return "%s <- (%s)" % (self.target, ", ".join(self.deps))


	def __str__(self):
		return self.toString()


	def __repr__(self):
		return self.toString()


class LinkRule(Rule):
	def __init__(self, deps, toolchain=None, target=None, forPython=True):
		super().__init__(
			[dep.target for dep in deps], toolchain, target, toolchain.pydext if forPython else toolchain.soext
		)


def extractCompilable(filenames):
	cext = {".c", ".cpp", ".cu"}
	return [filename for filename in filenames if any(filename.endswith(ext) for ext in cext)]


def validate(rules):
	errors = []
	cwd = os.path.commonprefix(list(itertools.chain(*(rule.deps for rule in rules))))

	for i, rule in enumerate(rules):
		print("### Validating rule '%s' (%s out of %s) ..." % (rule.target, i + 1, len(rules)), flush=True)

		try:
			deps = rule.toolchain.getDependencies(extractCompilable(rule.deps), cwd)

			if set(os.path.normcase(os.path.realpath(file)) for file in rule.deps) != deps:
				errors.append("Rule: %s\nValidated dependencies: (%s)\n" % (rule, ", ".join(deps)))

		except CompilerError as e:
			errors.append(str(e))

	if len(errors) > 0:
		raise BuildError("Dependencies validation failed with following error(s):\n\n%s" % "\n".join(errors))

	print("### Dependencies validation finished successfully", flush=True)


def clean(rules, linkrule):
	paths = set()

	for rule in rules + [linkrule]:
		removed = False

		try:
			path = os.path.dirname(rule.target)

			if path not in paths:
				paths.add(path)
				rule.toolchain.clearPath(path)

			os.remove(rule.target)
			removed = True

		except OSError:
			pass

		print("### %s '%s' ..." % ("Removed" if removed else "Skipped", rule.target), flush=True)


class Config:
	def __init__(self, rulesMap, toolchains):
		self.rulesMap, self.toolchains = rulesMap, toolchains


	def getSignature(self, rule):
		index = self.rulesMap.get(rule.target, None)

		if index is None:
			return None

		return self.toolchains[index]


	def save(self, filename):
		with open(filename, mode="w", encoding="utf-8") as f:
			config = {
				"rulesMap": self.rulesMap,
				"toolchains": self.toolchains
			}
			json.dump(config, f, indent=4)


	@classmethod
	def loadFromFile(cls, filename):
		rulesMap, toolchains = {}, []

		if os.path.exists(filename):
			with open(filename, mode="r", encoding="utf-8") as f:
				config = json.load(f)
				rulesMap, toolchains = config["rulesMap"], config["toolchains"]

		return cls(rulesMap, toolchains)


	@classmethod
	def loadFromRules(cls, rules):
		rulesMap, toolchains = {}, []
		ids = {}

		for rule in rules:
			toolchain = rule.toolchain
			index = ids.get(toolchain, None)

			if index is None:
				index = len(toolchains)

				toolchains.append(toolchain.signature())
				ids[toolchain] = index

			rulesMap[rule.target] = index

		return cls(rulesMap, toolchains)


def build(rules, linkrule, recompile=False, prevalidate=False):
	if prevalidate:
		validate(rules)

	configname = "%s.json" % linkrule.target

	prevcfg = Config.loadFromFile(configname)
	config = Config.loadFromRules(rules + [linkrule])

	errors = []
	needLinking = False

	try:
		for i, rule in enumerate(rules):
			needLinking = compileObj(rule, i, len(rules), recompile, errors, config, prevcfg) or needLinking

		if len(errors) > 0:
			raise BuildError("Build failed with following error(s):\n\n%s" % "\n".join(errors))

		link(linkrule, needLinking, config, prevcfg)
		print("### Build finished successfully", flush=True)

	finally:
		config.save(configname)


def compileObj(rule, i, nrules, recompile, errors, config, prevcfg):
	if recompile:
		info = "forcing recompilation"

	elif os.path.exists(rule.target):
		objMTime = os.path.getmtime(rule.target)
		changed = [dep for dep in rule.deps if os.path.getmtime(dep) > objMTime]

		if len(changed) > 0:
			info = "dependencies %s changed" % changed
		elif config.getSignature(rule) != prevcfg.getSignature(rule):
			info = "compiler settings changed"
		else:
			print("### Skipping building '%s' (%s out of %s) ..." % (rule.target, i + 1, nrules), flush=True)
			return False

	else:
		info = "output file not found"

	print("### Building '%s' - %s (%s out of %s) ..." % (rule.target, info, i + 1, nrules), flush=True)

	try:
		rule.toolchain.buildObject(rule.target, extractCompilable(rule.deps))

	except CompilerError as e:
		errors.append(str(e))

	return True


def link(rule, force, config, prevcfg):
	force = config.getSignature(rule) != prevcfg.getSignature(rule) or force

	if not os.path.exists(rule.target) or force:
		print("### Linking '%s' ..." % rule.target, flush=True)
		rule.toolchain.link(rule.target, rule.deps)

	else:
		print("### Skipping linking '%s' ..." % rule.target, flush=True)


def unittest():
	header = """
#pragma once
PyObject *hello(PyObject *self, PyObject *args);
"""

	body1 = """
#include <Python.h>
#include "header.h"


static PyMethodDef methods[] = {
	{"hello", hello, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};


static PyModuleDef mod = {
	PyModuleDef_HEAD_INIT,
	.m_name = "test",
	.m_methods = methods
};


PyMODINIT_FUNC PyInit_test(void)
{
	return PyModule_Create(&mod);
}
"""

	body2 = """
#include <Python.h>
#include "header.h"


PyObject *hello(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	puts("Hello, Build!");
	fflush(stdout);

	Py_RETURN_NONE;
}
"""

	with open("./TestData/header.h", mode="w", encoding="utf-8") as f:
		f.write(header)

	srcnames = ["./TestData/test.c", "./TestData/test2.c"]

	for srcname, src in zip(srcnames, [body1, body2]):
		with open(srcname, mode="w", encoding="utf-8") as f:
			f.write(src)

	toolchain = guessToolchain(verbose=2).withOptimizationLevel(level=4)

	rules = [Rule([srcname, "./TestData/header.h"], toolchain=toolchain) for srcname in srcnames]
	linkrule = LinkRule(rules, toolchain=toolchain)

	validate(rules)
	build(rules, linkrule)

	module = loadDynamicModule(os.path.join(os.path.dirname(__file__), linkrule.target))

	try:
		module.hello()

	finally:
		clean(rules, linkrule)


if __name__ == "__main__":
	unittest()
