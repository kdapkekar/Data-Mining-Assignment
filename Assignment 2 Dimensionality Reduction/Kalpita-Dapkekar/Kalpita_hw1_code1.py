import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics
from csv import reader
import seaborn as sns

def center(data):
    dataset = scale(data, axis = 0, with_mean=True, with_std=False)
    return dataset

def cov1(data1):
    covariance = np.cov(data1, rowvar=False, bias=True)
    return covariance

def cov2(data2):
    inner_product = np.dot(np.transpose(data2), data2)
    Cov_matrix = inner_product/6
    print(inner_product.size)
    return Cov_matrix

def cov3(data3):
    n=len(data3)
    sum1 = 0
    for i in range(0,n):
        sum1 = sum1 + np.outer(np.transpose(data3[i]),data3[i])
    return sum1/6


def noOfPrincipalComponantsR(eigenValues, alpha):
    index = eigenValues.argsort()[::-1]

    cumulativeSumList = list()
    summ = 0
    for ev in eigenValues:
        summ = summ + ev
        cumulativeSumList.append(summ)

    rList = list()
    for element in cumulativeSumList:
        fraction = (element / summ) * 100
        rList.append(fraction)

    for fraction in rList:
        if fraction > alpha:
            indexOfR = rList.index(fraction)
            break

    print("rList")
    print(rList)
    return indexOfR + 1

def ReduceDimension(dataset, eigenVectors):
    reducedMatrix = np.dot(dataset, eigenVectors.T)
    return reducedMatrix

def plotLines(reduced_dim_data_mat):
    x1= list(range(1,3))
    y1= reduced_dim_data_mat[0]
    plt.plot(x1, y1, label= "PC1")
    x2= x1
    y2= reduced_dim_data_mat[1]
    plt.plot(x2, y2, label = "PC2")
    plt.title('Vectors PC1 and PC2 on graph')
    plt.legend()
    plt.show()



def plotScatter(reduced_dim_data_mat):
    area = np.pi * 3
    for i in range(0,len(reduced_dim_data_mat)):
        x = reduced_dim_data_mat[i][0]
        y = reduced_dim_data_mat[i][1]
        plt.scatter(x, y, s = area, c = '#007f40', alpha = 0.2)
    plt.title("Scatter Plot for first two PCs")
    #plt.scatter(x, y)
    plt.show()

if __name__=='__main__':
    #data = pd.read_csv("hw1Data(1).csv")
    textcsv = "hw1Data(1).csv"
    data = np.genfromtxt(textcsv, delimiter=',')
    #(A) Center Data
    Centered_data = center(data)
    print Centered_data


    #(B) Find Covarience Matrix in three different ways
    start_time = time.time()
    Cov_numpy = cov1(Centered_data)                                     # using numpy.cov() function
    print("%s seconds" % (time.time() - start_time))
    print(Cov_numpy)
    start_time = time.time()
    Cov_inner = cov2(Centered_data)                                     #using inner product
    print("%s seconds" % (time.time() - start_time))
    print('Covariance using Inner Product',Cov_inner)
    start_time = time.time()
    Cov_outer = cov3(Centered_data)                                     #using outer product
    print("%s seconds" % (time.time() - start_time))
    print('Covariance using outer Product',Cov_outer)

    #(C) Eigenvectors and Eigenvalues
    eigen_Val, eigen_Vect = np.linalg.eig(Cov_numpy)
    print("Eigen Values")
    print(eigen_Val)
    print("Eigen Vectors")
    print(eigen_Vect)

    #(D) Find Number of Principal Components r
    num_of_components = noOfPrincipalComponantsR(eigen_Val, 90)
    print('Number of Components',num_of_components)
    #plotScatter(X_pca)
    #(E) Reduced dimension data matrix with first Two PCs
    reduced_dim_data_mat = ReduceDimension(Centered_data, eigen_Vect[:2])
    print("reducedMatrix")
    print(reduced_dim_data_mat)
    plotLines(reduced_dim_data_mat)
    f = open('components.txt', 'w')
    f.writelines(["%s," % eigen_Vect[0]])
    f.writelines("\n")
    f.writelines(["%s," % eigen_Vect[1]])
    #(F) Reduced dimension data matrix with first Two PCs
    plotScatter(reduced_dim_data_mat)