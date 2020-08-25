import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers import Sequential
from PuzzleLib.Modules import Conv2D, AvgPool2D, BatchNorm2D, Activation, relu, Flatten
from PuzzleLib.Cost import BCE


def buildNet():
	net = Sequential(name="test-net")

	net.append(Conv2D(1, 2, 3, wscale=1.0, initscheme="gaussian"))
	net.append(AvgPool2D(2, 2))

	net.append(BatchNorm2D(2))
	net.append(Activation(relu))

	net.append(Conv2D(2, 1, 2, wscale=1.0, initscheme="gaussian"))
	net.append(Flatten())

	return net


def gradientCheck(mod, data, target, cost, h=1e-3):
	vartable = mod.getVarTable()

	mod(data)
	error, grad = cost(mod.data, target)
	mod.backward(grad, updGrad=False)

	for var in vartable.keys():
		w = var.data.get()
		dw = -var.grad.get()

		for i in range(w.ravel().shape[0]):
			wph = np.copy(w)
			wmh = np.copy(w)

			wph.ravel()[i] = w.ravel()[i] + h
			var.data.set(wph)
			yph, _ = cost(mod(data), target)

			wmh.ravel()[i] = w.ravel()[i] - h
			var.data.set(wmh)
			ymh, _ = cost(mod(data), target)

			host = (yph - ymh) / (2.0 * h)
			dev = dw.ravel()[i]
			var.data.set(w)

			print(abs((host - dev) / (dev + h)))


def main():
	net = buildNet()
	cost = BCE()

	data = gpuarray.to_gpu(np.random.randn(1, 1, 6, 6).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randint(0, 2, size=(1, )))

	gradientCheck(net, data, target, cost)


if __name__ == "__main__":
	main()
