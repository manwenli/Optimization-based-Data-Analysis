import numpy as np
import scipy as sp
from mnist_tools import *
from plot_tools import *
from nearest_neighbors import *

def main():
    #read in train and test data
    datafile = "mnist_all.mat"
    train = load_train_data(datafile)
    Test,testLabels = load_test_data(datafile)
    
    #cast A
    A = train[0]
    for i in range(9):
        temp = train[i+1].astype(float)
        #temp = train[i]
        A = np.concatenate((A,temp),axis = 0)   
    
    test = Test.astype(float) 
    
    #part(c)(i)
    #AT: column vector is an image
    AT = A.transpose()
    testT = test.transpose()
    avg = AT.mean(axis = 1)
    #tavg = testT.mean(axis = 1)
    #transpose mean into row vector
    avgT = avg.transpose()
    #tavgT = tavg.transpose()
    
    #centered data where each row is an image
    for k in range(A.shape[0]):
        A[k,:]=A[k,:]-avgT
    
    for k in range(test.shape[0]):
        test[k,:]=test[k,:]-avgT
    
    
    
    #now each column is an image
    C = A.transpose()
    
    #perform SVD,singular value should be U
    U,S,Vh = sp.linalg.svd(C,full_matrices = False)
    #plot sigma k vs. k
    K = np.arange(len(S))
    plt.figure()
    plt.scatter(K,S)
    plt.show()
    
    #(ii) plot singular values
    imgs =[]
    for i in range(10):
        imgs.extend([U[:,i].transpose()])
    plot_image_grid(imgs,"Top 10 Singular Values")
    
    
    #(iii) PCA
    #1. determing smallest k
    s = S[:100]
    k = np.arange(100)
    plt.figure()
    plt.scatter(k,s)
    plt.show()
    
    #2. choose k = 50, choosing the first 8 vectors from U, then creating projection matrix uu^T
    u = U[:,:8]
    p = np.matmul(u,u.transpose())
    #project train data
    train_proj = (np.matmul(p, C)).transpose()
    #project test data 
    test_proj = (np.matmul(p,test.transpose())).transpose()
    
    #3. run nearest neightbor for test
    newTrain = []
    #for n in range(10):
        #x = train_proj[n*100:n+100,:]
        #newTrain.extend([x])
    x1 = train_proj[0:100,:]
    x2 = train_proj[100:200,:]
    x3 = train_proj[200:300,:]
    x4 = train_proj[300:400,:]
    x5 = train_proj[400:500,:]
    x6 = train_proj[500:600,:]
    x7 = train_proj[600:700,:]
    x8 = train_proj[700:800,:]
    x9 = train_proj[800:900,:]
    x10 = train_proj[900:1000,:]
    
    newTrain = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    
    imgs = []
    estLabels = []
    for i in range(len(testLabels)) :
        trueDigit = testLabels[i]
        testImage = test_proj[i,:]
        testImage_original = Test[i,:]
        #(nnDig,nnIdx) = compute_nearest_neighbors(train,testImage)
        (nnDig,nnIdx) = compute_nearest_neighbors(newTrain,testImage)
        #imgs.extend( [testImage,train_proj[(nnDig-1)*100+nnIdx]])
        imgs.extend( [testImage_original,train[nnDig][nnIdx,:]])
        estLabels.append(nnDig)

    row_titles = ['Test','Nearest']
    col_titles = ['%d vs. %d'%(i,j) for i,j in zip(testLabels,estLabels)]
    plot_image_grid(imgs,
                    "Image-NearestNeighbor",
                    (28,28),len(testLabels),2,True,row_titles=row_titles,col_titles=col_titles)
    

if __name__ == "__main__" :
    main()
    
    
#Answer for 6 c) iv:
#In the full training set, many features measure the related properties of the image and thus are redundant. PCA performs a linear transformation to move the original set of features to a lower dimensional space composed by principal component. By performing PCA, we don't need to deal with the "redundancy" that are low variant when performing the classification algorithms. We only deal with the high variance which will generate a more accurate "split" between classes. 
#Reference: A Medium Corporation





    