import numpy as np

def loadData(dataFileName):
	inputMatrix = np.loadtxt(open(dataFileName,'rb'), delimiter=",", skiprows=1)  
	return inputMatrix

def preProcessing(inputMatrix):
	arrayInput = np.array(inputMatrix)
	print arrayInput.shape
	numSample, numFeature = arrayInput.shape
	numFeature = numFeature-1
	labelArray = arrayInput[:,-1]
	featureArray = arrayInput[:, 0:numFeature]
	return labelArray, featureArray

dataFileName = "data/data.csv"
inputMatrix = loadData(dataFileName)
print len(inputMatrix)
labelArray, featureArray = preProcessing(inputMatrix)
