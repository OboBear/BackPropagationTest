# -*- coding:utf-8 -*-
import numpy as np
import math
import random



def logistic(x):
	return 1.0/(1+np.exp(-x))

class NeuronNetwork(object):
	def __init__(self, inputSize, hideSize, outputSize):
		print("\n=============================")
		print("Init")
		print("=============================")

		print("NeuronNetwork inputSize = %d hideSize = %d outputSize = %d" %(inputSize, hideSize, outputSize))
		self.hideLayer = NeuronLayer(hideSize, inputSize)
		self.outputLayer = NeuronLayer(outputSize, hideSize)

	def appendOne(self, npArray):
		array = list(npArray)
		array.append(1)
		return np.array(array)


	def train(self, input, targetOutput):
		print("\n=============================")
		print("Train")
		print("=============================")

		input = appendOne(input)
		hideOutput = self.hideLayer.calculate(input);
		hideOutput = list(hideOutput)
		hideOutput.extends([1])
		hideOutput = np.array(hideOutput)

		hideOutput = appendOne(hideOutput)
		print("=============================")
		print("HideOutput")
		print(hideOutput)
		output = self.outputLayer.calculate(hideOutput);
		print("=============================")
		print("Output:")
		print(output)
		print("=============================")
		print("Back")
		differ = targetOutput - output
		print(differ * differ)
		thita = differ * output * (1 - output) * hideOutput;





class NeuronLayer(object):
	def __init__(self, layerSize, inputSize):
		self.bies = 1
		self.neuronLayer = []
		for x in range(0,layerSize):
			self.neuronLayer.append(Neuron(inputSize + 1))
		print("NeuronLayer inputSize = %d bias = %d" %(inputSize, 1))

	def calculate(self, input):
		result = []
		print("self.neuronLayer")
		print(len(self.neuronLayer))
		for x in range(0, len(self.neuronLayer)):
			result.append(self.neuronLayer[x].calculate(input));
		print("result")
		print(result)
		print(np.array(result))
		return result


class Neuron(object):
	def __init__(self, inputSize):
		print("int Neuron")
		self.inputSize = inputSize
		self.inputWight = np.ones(inputSize)

	def calculate(self, inputData):
		print("inputData")
		print(inputData)
		self.netOut = np.sum(inputData * self.inputWight)
		self.netOut = logistic(self.netOut)
		return self.netOut

neuronNetwork = NeuronNetwork(inputSize = 2, hideSize = 2,outputSize = 1)
neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1]))

# [0,0],[0]
# [0,1],[1]
# [1,0],[1]
# [1,1],[0]
#


