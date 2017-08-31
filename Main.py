# -*- coding:utf-8 -*-
import numpy as np
import math
import random

# 学习率
LEARNING_RATE = 0.5


def logistic(x):
	return 1.0 / (1 + np.exp(-x))

def log(output):
	1+1
	# print(output)

# 神经网络
class NeuronNetwork(object):
	def __init__(self, inputSize, net1Size, net2Size):
		log("\n=============================")
		log("Init")
		log("=============================")

		log("NeuronNetwork inputSize = %d net1Size = %d net2Size = %d" %(inputSize, net1Size, net2Size))
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
		log("\n=============================")
		log("Train")
		log("=============================")
		# 添加偏置 1
		net1Input = self.appendOne(input)
		net1Output = self.net1Layer.calculate(net1Input);
		net1Output = np.array(net1Output)

		# 添加偏置 1
		net1Output = self.appendOne(net1Output)
		log("=============================")
		log("Calculate net1Output")
		log(net1Output)
		net2Output = self.net2Layer.calculate(net1Output);
		net2Output = np.array(net2Output)
		log("=============================")
		log("Calculate TotalOutput")
		log(net2Output)

		# 开始反向传播调整权值
		log("=============================")
		log("Calculate distance")
		temp1 = (net2Output - targetOutput) * net2Output * (1 - net2Output)
		temp1.shape = (len(temp1), 1)
		log(temp1)
		net1Output.shape = (1, len(net1Output))
		# 生成第二层的梯度
		thetaOutput = np.dot(temp1, net1Output)
		log("=============================")
		log("thetaOutput")
		log(thetaOutput)
		print(thetaOutput)
		log("=============================")

		net2Weight = self.net2Layer.getWeight()
		log("net2Weight")
		log(net2Weight)

		temp1.shape = (1, len(temp1))
		net1InputMatrix = np.array([net1Input])
		thetaHideStep1 = np.dot((np.dot(temp1,net2Weight) * net1Output * (1 - net1Output)).T , net1InputMatrix)

		# print(thetaHideStep1)
		# thetaHideStep1 = np.split(thetaHideStep1,[self.net1Size])[0]
		log(thetaHideStep1)
		# print(thetaHideStep1)
		# thetaHideStep1.shape = [len(thetaHideStep1), 1]
		# thetaHideStep1 = np.array(thetaHideStep1)
		newInput = np.array([net1Input])
		log("newInput")
		log(newInput)
		# print(newInput)
		log("=============================")
		log("thetaHideStep1")
		log(thetaHideStep1)
		# 生成第一层梯度
		thetaHide = thetaHideStep1 * newInput

		log(thetaHide)
		# print(thetaHide)

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
		log("NeuronLayer inputSize = %d bias = %d" %(inputSize, 1))

	def calculate(self, input):
		result = []
		log("self.neuronLayer")
		log(len(self.neuronLayer))
		for x in range(0, len(self.neuronLayer)):
			result.append(self.neuronLayer[x].calculate(input));
		log("result")
		log(result)
		log(np.array(result))
		return result
	def updateWeight(self, theta):
		for x in xrange(0,len(self.neuronLayer)):
			self.neuronLayer[x].updateWeight(theta[x])
	def getWeight(self):
		weight = []
		for x in xrange(0, len(self.neuronLayer)):
			weight.append(self.neuronLayer[x].getWeight())
		return np.array(weight)

# 神经节点
class Neuron(object):
	def __init__(self, inputSize, weight = None):
		log("int Neuron")
		self.inputSize = inputSize
		if weight is not None:
			self.inputWight = weight
		else:
			self.inputWight = np.ones(inputSize)
	
	def calculate(self, inputData):
		log("inputData")
		log(inputData)
		log(self.inputWight)
		self.netOut = np.sum(inputData * self.inputWight)
		self.netOut = logistic(self.netOut)
		return self.netOut

	def updateWeight(self, thetaX):
		self.inputWight = self.inputWight - LEARNING_RATE * thetaX;

	def getWeight(self):
		return self.inputWight

neuronNetwork = NeuronNetwork(inputSize = 2, net1Size = 6,net2Size = 1)
# for x in xrange(1,1000):
# 	neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1, 2]))	
for x in xrange(0,100):
	neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([0]))
	# neuronNetwork.train(input = np.array([0,1]), targetOutput = np.array([0]))
	# neuronNetwork.train(input = np.array([1,1]), targetOutput = np.array([1]))
	# neuronNetwork.train(input = np.array([0,0]), targetOutput = np.array([1]))
# neuronNetwork.train(input = np.array([1,0]), targetOutput = np.array([1, 2]))

log("Train result")
outputFinal = neuronNetwork.calculate(input = np.array([1,0]))
log("outputFinal1")
log(outputFinal)
# print(outputFinal)

outputFinal = neuronNetwork.calculate(input = np.array([0,0]))
log("outputFinal2")
log(outputFinal)
# print(outputFinal)

log("Train End")

# [0,0],[0]
# [0,1],[1]
# [1,0],[1]
# [1,1],[0]
#
