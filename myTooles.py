import numpy
def loadData(dataFileName):
	inputMatrix = numpy.loadtxt(open(dataFileName,'rb'), delimiter=",", skiprows=1)  
	return inputMatrix


dataFileName = "data/data.csv"
inputMatrix = loadData(dataFileName)
#labelMatrix = []
#for i in range(0, len(inputMatrix)):
#	labelMatrix = inputMatrix[i][-1]
#print labelMatrix
labelMatrix = inputMatrix[:][-1]
#print len(labelMatrix)
print len(inputMatrix)
print inputMatrix[1]
#print labelMatrix[279:319]