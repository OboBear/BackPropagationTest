# -*- coding:utf-8 -*-
import numpy as np
import math
import random

# 学习率
LEARNING_RATE = 0.5


def logistic(x):
	return 1.0/(1+np.exp(-x))

# 神经网络
class NeuronNetwork(object):
	def __init__(self, inputSize, net1Size, net2Size):
		print("\n=============================")
		print("Init")
		print("=============================")

		print("NeuronNetwork inputSize = %d net1Size = %d net2Size = %d" %(inputSize, net1Size, net2Size))
		self.inputSize = inputSize
		self.net1Size = net1Size
		self.net2Size = net2Size
		self.net1Layer = NeuronLayer(net1Size, inputSize)
		self.net2Layer = NeuronLayer(net2Size, net1Size)

	def appendOne(self, npArray):
		array = list(npArray)
		array.append(1)
		return np.array(array)

	def train(self, input, targetOutput):
		print("\n=============================")
		print("Train")
		print("=============================")
		# 添加偏置 1
		input = self.appendOne(input)
		hideOutput = self.net1Layer.calculate(input);
		hideOutput = np.array(hideOutput)

		# 添加偏置 1
		hideOutput = self.appendOne(hideOutput)
		print("=============================")
		print("Calculate HideOutput")
		print(hideOutput)
		output = self.net2Layer.calculate(hideOutput);
		output = np.array(output)
		print("=============================")
		print("Calculate TotalOutput")
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
		weight = self.net2Layer.getWeight()
		# print("weight")


		temp1.shape = (1, len(temp1))
		# print("temp1*weight")
		# print(temp1)
		# print(np.dot(temp1,weight))
		thetaHideStep1 = np.dot(temp1,weight)[0] * input
		thetaHideStep1 = np.split(thetaHideStep1,[self.net1Size])[0]
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

		self.net2Layer.updateWeight(thetaOutput)
		self.net1Layer.updateWeight(thetaHide)

	def calculate(self, input):
		net1Input = self.appendOne(input)
		net1Output = self.net1Layer.calculate(net1Input);
		net2Input = self.appendOne(net1Output)
		net2Output = self.net2Layer.calculate(net2Input)
		return net2Output

# 单层网络
class NeuronLayer(object):
	def __init__(self, layerSize, inputSize):
		self.bies = 1
		self.neuronLayer = []
		for x in range(0,layerSize):
			self.neuronLayer.append(Neuron(inputSize = inputSize + 1))
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

# 神经节点
class Neuron(object):
	def __init__(self, inputSize, weight = None):
		print("int Neuron")
		self.inputSize = inputSize
		if weight is not None:
			self.inputWight = weight
		else:
			self.inputWight = np.ones(inputSize)
	
	def calculate(self, inputData):
		print("inputData")
		print(inputData)
		print(self.inputWight)
		self.netOut = np.sum(inputData * self.inputWight)
		self.netOut = logistic(self.netOut)
		return self.netOut

	def updateWeight(self, thetaX):
		self.inputWight = self.inputWight + LEARNING_RATE * thetaX;

	def getWeight(self):
		return self.inputWight

neuronNetwork = NeuronNetwork(inputSize = 2, net1Size = 2,net2Size = 2)
# for x in xrange(1,1000):
# 	neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1, 2]))	
for x in xrange(0,100):
	neuronNetwork.train(input = np.array([0.05,0.10]), targetOutput = np.array([0.01, 0.99]))
	neuronNetwork.train(input = np.array([0.3,0.70]), targetOutput = np.array([0.90, 0.01]))
# neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1, 2]))

print("Train result")
outputFinal = neuronNetwork.calculate(input = np.array([0.05,0.10]))
print("outputFinal1")
print(outputFinal)

outputFinal = neuronNetwork.calculate(input = np.array([0.3,0.70]))
print("outputFinal2")
print(outputFinal)

print("Train End")

# [0,0],[0]
# [0,1],[1]
# [1,0],[1]
# [1,1],[0]
#
