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
		self.inputSize = inputSize
		self.hideSize = hideSize
		self.outputSize = outputSize
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

		input = self.appendOne(input)
		hideOutput = self.hideLayer.calculate(input);
		hideOutput = np.array(hideOutput)

		hideOutput = self.appendOne(hideOutput)
		print("=============================")
		print("Calculate HideOutput")
		print(hideOutput)
		output = self.outputLayer.calculate(hideOutput);
		output = np.array(output)
		print("=============================")
		print("Calculate TotalOutput:")
		print(output)
		print("=============================")
		print("Calculate distance")
		differ = targetOutput - output
		print(differ * differ)
		print("temp1")
		temp1 = differ * output * (1 - output);
		temp1.shape = (len(temp1), 1)
		print(temp1)
		hideOutput.shape = (1, len(hideOutput))
		thetaOutput = np.dot(temp1, hideOutput)
		print("=============================")
		print("thetaOutput")
		print(thetaOutput)

		# temp1
		weight = self.outputLayer.getWeight()
		# print("weight")
		# print(weight)


		temp1.shape = (1, len(temp1))
		# print("temp1*weight")
		# print(temp1)
		# print(np.dot(temp1,weight))
		thetaHideStep1 = np.dot(temp1,weight)[0] * input
		thetaHideStep1 = np.split(thetaHideStep1,[self.hideSize])[0]
		print(thetaHideStep1)
		thetaHideStep1.shape = [len(thetaHideStep1), 1]
		thetaHideStep1 = np.array(thetaHideStep1)
		newInput = np.array([input])
		print("newInput")
		print(newInput)
		print("=============================")
		print("thetaHideStep1")
		print(thetaHideStep1)
		thetaHide = np.dot(thetaHideStep1, newInput)
		print(thetaHide)

		self.outputLayer.updateWeight(thetaOutput)
		self.hideLayer.updateWeight(thetaHide)


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
	def updateWeight(self, theta):
		for x in xrange(0,len(theta)):
			self.neuronLayer[x].updateWeight(theta[x])
	def getWeight(self):
		weight = []
		for x in xrange(0, len(self.neuronLayer)):
			weight.append(self.neuronLayer[x].getWeight())
		return np.array(weight)


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

	def updateWeight(self, thetaX):
		self.inputWight = self.inputWight + 0.5 * thetaX;

	def getWeight(self):
		return self.inputWight

neuronNetwork = NeuronNetwork(inputSize = 2, hideSize = 2,outputSize = 2)
for x in xrange(1,1000):
	neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1, 2]))	


# [0,0],[0]
# [0,1],[1]
# [1,0],[1]
# [1,1],[0]
#


