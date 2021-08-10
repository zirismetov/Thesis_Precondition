import numpy as np
matrixA = [[2,3,4],[1,2,3]]
matrixB = [[3,1] ,[1,1],[5,5]]
rowA = len(matrixA)
colA = len(matrixA[0])
rowB = len(matrixB)
colB = len(matrixB[0])

result = [[0 for p in range(colB)] for q in range(rowA)]

def getColAsList(matrixToManipulate, col):
    myList = []
    numOfRows = len(matrixToManipulate)
    for i in range(numOfRows):
        myList.append(matrixToManipulate[i][col])
    return myList

def getCell(matrixA, matrixB, r, c):
    matrixBCol = getColAsList(matrixB, c)
    lenOfList = len(matrixBCol)
    productList = [matrixA[r][i]*matrixBCol[i] for i in range(lenOfList)]
    return sum(productList)

def colaculate(matrixA, matrixB):
    if colA != rowB:
        print('The two matrices cannot be multiplied')
    else:
        print('\nThe result is')
        for i in range(rowA):
            for j in range(colB):
                result[i][j] = getCell(matrixA, matrixB, i, j)
            print(result[i])

