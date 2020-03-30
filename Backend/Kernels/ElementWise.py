from PuzzleLib import Config


sigmoidKer = None
sigmoidDerKer = None
tanhKer = None
tanhDerKer = None
reluKer = None
reluDerKer = None
leakyReluKer = None
leakyReluDerKer = None
eluKer = None
eluDerKer = None
softPlusKer = None
softPlusDerKer = None
clipKer = None
clipDerKer = None
geluKer = None
geluDerKer = None

dropoutKer = None
dropout2dKer = None
rbmKer = None
absKer = None
toVectorAddVectorKer = None

classicMomSGDKer = None
nesterovMomSGDKer = None
rmspropKer = None
adamKer = None
rmspropGravesKer = None
adagradKer = None
adadeltaKer = None
smorms3Ker = None

weightDecayKer = None
linearKer = None
addKer = None
mulKer = None
l1penaltyKer = None
l1gradKer = None

castFP16toFP32 = None
castFP32toFP16 = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend
	initGPU(Backend)


def initHip():
	from PuzzleLib.Hip import Backend
	initGPU(Backend)


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=2)

	global sigmoidKer, sigmoidDerKer, tanhKer, tanhDerKer, reluKer, reluDerKer, leakyReluKer, leakyReluDerKer
	global eluKer, eluDerKer, softPlusKer, softPlusDerKer, clipKer, clipDerKer, geluKer, geluDerKer
	sigmoidKer = backend.sigmoidKer
	sigmoidDerKer = backend.sigmoidDerKer
	tanhKer = backend.tanhKer
	tanhDerKer = backend.tanhDerKer
	reluKer = backend.reluKer
	reluDerKer = backend.reluDerKer
	leakyReluKer = backend.leakyReluKer
	leakyReluDerKer = backend.leakyReluDerKer
	eluKer = backend.eluKer
	eluDerKer = backend.eluDerKer
	softPlusKer = backend.softPlusKer
	softPlusDerKer = backend.softPlusDerKer
	clipKer = backend.clipKer
	clipDerKer = backend.clipDerKer
	geluKer = backend.geluKer
	geluDerKer = backend.geluDerKer

	global dropoutKer, dropout2dKer, rbmKer, absKer, toVectorAddVectorKer
	dropoutKer = backend.dropoutKer
	dropout2dKer = backend.dropout2dKer
	rbmKer = backend.rbmKer
	absKer = backend.absKer
	toVectorAddVectorKer = backend.toVectorAddVectorKer

	global classicMomSGDKer, nesterovMomSGDKer, rmspropKer, adamKer, rmspropGravesKer, adagradKer, adadeltaKer
	global smorms3Ker
	classicMomSGDKer = backend.classicMomSGDKer
	nesterovMomSGDKer = backend.nesterovMomSGDKer
	rmspropKer = backend.rmspropKer
	adamKer = backend.adamKer
	rmspropGravesKer = backend.rmspropGravesKer
	adagradKer = backend.adagradKer
	adadeltaKer = backend.adadeltaKer
	smorms3Ker = backend.smorms3Ker

	global weightDecayKer, linearKer, addKer, mulKer, l1penaltyKer, l1gradKer
	weightDecayKer = backend.weightDecayKer
	linearKer = backend.linearKer
	addKer = backend.addKer
	mulKer = backend.mulKer
	l1penaltyKer = backend.l1penaltyKer
	l1gradKer = backend.l1gradKer

	global castFP16toFP32, castFP32toFP16
	castFP16toFP32 = backend.castFP16toFP32
	castFP32toFP16 = backend.castFP32toFP16


def initCPU():
	from PuzzleLib.CPU.Kernels import ElementWise

	global sigmoidKer, sigmoidDerKer, tanhKer, tanhDerKer, reluKer, reluDerKer, leakyReluKer, leakyReluDerKer
	global eluKer, eluDerKer, softPlusKer, softPlusDerKer, clipKer, clipDerKer
	sigmoidKer = ElementWise.sigmoidKer
	sigmoidDerKer = ElementWise.sigmoidDerKer
	tanhKer = ElementWise.tanhKer
	tanhDerKer = ElementWise.tanhDerKer
	reluKer = ElementWise.reluKer
	reluDerKer = ElementWise.reluDerKer
	leakyReluKer = ElementWise.leakyReluKer
	leakyReluDerKer = ElementWise.leakyReluDerKer
	eluKer = ElementWise.eluKer
	eluDerKer = ElementWise.eluDerKer
	softPlusKer = ElementWise.softPlusKer
	softPlusDerKer = ElementWise.softPlusDerKer
	clipKer = ElementWise.clipKer
	clipDerKer = ElementWise.clipDerKer

	global dropoutKer, dropout2dKer, rbmKer, absKer, toVectorAddVectorKer
	dropoutKer = ElementWise.dropoutKer
	dropout2dKer = ElementWise.dropout2dKer
	rbmKer = ElementWise.rbmKer
	absKer = ElementWise.absKer
	toVectorAddVectorKer = ElementWise.toVectorAddVectorKer

	global classicMomSGDKer, nesterovMomSGDKer, rmspropKer, adamKer, rmspropGravesKer, adagradKer, adadeltaKer
	global smorms3Ker
	classicMomSGDKer = ElementWise.classicMomSGDKer
	nesterovMomSGDKer = ElementWise.nesterovMomSGDKer
	rmspropKer = ElementWise.rmspropKer
	adamKer = ElementWise.adamKer
	rmspropGravesKer = ElementWise.rmspropGravesKer
	adagradKer = ElementWise.adagradKer
	adadeltaKer = ElementWise.adadeltaKer
	smorms3Ker = ElementWise.smorms3Ker

	global weightDecayKer, linearKer, addKer, mulKer, l1penaltyKer, l1gradKer
	weightDecayKer = ElementWise.weightDecayKer
	linearKer = ElementWise.linearKer
	addKer = ElementWise.addKer
	mulKer = ElementWise.mulKer
	l1penaltyKer = ElementWise.l1penaltyKer
	l1gradKer = ElementWise.l1gradKer


autoinit()
