import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.stats import pearsonr

def loadData(dataFileName):
	inputMatrix = np.loadtxt(open(dataFileName,'rb'), delimiter=",", skiprows=1)  
	return inputMatrix

def preProcessing(inputMatrix):
	arrayInput = np.array(inputMatrix)
	numSample, numFeature = arrayInput.shape
	numFeature = numFeature-1
	labelArray = arrayInput[:,-1]
	featureArray = arrayInput[:, 0:numFeature]
	return labelArray, featureArray

def getPosNegIdx(label):
	posIdx = []
	negIdx = []
	for i in range(0, len(label)):
		if label[i]==0:
			negIdx.append(i)
		else:
			posIdx.append(i)
	return posIdx, negIdx

def selectOneFeature(featureIndex, featureSpace):
	return featureSpace[:,featureIndex]

def getfStatis(featureIdx, X_train, posIdx, negIdx):
	featureSelectedi = selectOneFeature(featureIdx, X_train)
	nPos = len(posIdx)
	nNeg = len(negIdx)
	sigmaSQPos = np.var(featureSelectedi[posIdx])
	sigmaSQPos = np.asarray(sigmaSQPos)
	sigmaSQNeg = np.var(featureSelectedi[negIdx])
	sigmaSQNeg = np.asarray(sigmaSQNeg)
	nPosArray = np.asarray(nPos)
	nNegArray = np.asarray(nNeg)
	sigmaSQ = [(nPosArray-1)*sigmaSQPos+(nNegArray-1)*sigmaSQNeg]/(nPosArray+nNegArray-2)
	gMean = np.mean(featureSelectedi)
	gPosMean = np.mean(featureSelectedi[posIdx])
	gNegMean = np.mean(featureSelectedi[negIdx])
	outputTest.write("gPosMean = %f, gMean = %f, gNegMean = %f, sigmaSQ = %f\n"% (gPosMean, gMean, gNegMean, sigmaSQ))
	if sigmaSQ == 0:
		fStatis = 0
	else:
		fStatis = [nPosArray*(gPosMean-gMean)*(gPosMean-gMean)+nNegArray*(gNegMean-gMean)*(gNegMean-gMean)]/(sigmaSQ)
#	print fStatis
	return fStatis

def findFirstFeature(trainFStatis):
	#just use the maxF-statis as the first feature
#	return trainFStatis.max(), trainFStatis.argmax() 
	return trainFStatis.argmax()

def crossValidation(featureArray, labelArray, test_size):
	X_train, X_test, y_train, y_test = train_test_split(featureArray, labelArray, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test
	w
crossValRatio = 0.2
#load the data, get the features and label
dataFileName = "data/data.csv"
inputMatrix = loadData(dataFileName)
labelArray, featureArray = preProcessing(inputMatrix)
#split the data
X_train, X_test, y_train, y_test = crossValidation(featureArray, labelArray, crossValRatio)
sampleTrainNum, featureTrainNum = X_train.shape
posIdx, negIdx = getPosNegIdx(y_train)
featureIdxi = 1
featureIdxj = 1
#f(i, h) = featureSelectedi, c(i,j) = PCCij
#print getfStatis(featureIdxi, X_train, posIdx, negIdx)
#print PCCij
featureSelectedi = selectOneFeature(featureIdxi, X_train)
PCCij = pearsonr(selectOneFeature(featureIdxi, X_train), selectOneFeature(featureIdxj, X_train))[0]
outputTest = open("outputTest.txt", "w")
trainFStatis = np.zeros(featureTrainNum)
for i in range(0, featureTrainNum):
	trainFStatis[i] = getfStatis(i, X_train, posIdx, negIdx)

seletedFeature = set()
setUniversalSet =  { x for x in range(featureTrainNum) }

findMaxFeature = findFirstFeature(trainFStatis)

seletedFeature = seletedFeature.add(findMaxFeature)


#trainFStatis.append(
#print trainFStatis
#print max(trainFStatis)


