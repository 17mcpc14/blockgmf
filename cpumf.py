import numpy as np
import glob
from os.path import exists, basename
import sys
import time
import math
import numpy as np

#sys.stdout = open('cpumf.out', 'w+')

def error(R, P , Q):
    Rc = np.dot(P,Q)
    Rc = Rc - R
    Rc = np.square(Rc)
    error = np.sum(Rc)/10000
    return np.sqrt(error)

def matrix_factorization(R, K=10, steps=10, alpha=0.0001, beta=0.01):
    
    t0 = time.clock()

    P, Q, Rc = np.ones((R.shape[0], K)), np.ones((K, R.shape[1])), np.zeros((R.shape[0], R.shape[1])),  #initPQ(R.shape[0], K, R.shape[1])
    x, y1, y2 = [], [], []
    
    for step in range(steps):
        
        print("Step : ", step)

        for i , j in np.ndindex(R.shape):
            eij = R[i,j] - np.dot(P[i,:], Q[:,j])
        
            for k in range(K):
                P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
                Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
        e = 0
        Rc = np.dot(P, Q)
        for i , j in np.ndindex(R.shape):
            e = e + pow(R[i,j] - Rc[i,j] , 2)
            for k in range(K):
                e = e + (beta/2) * ( pow(P[i,k],2) + pow(Q[k,j],2) )
        
        print("Time till now :", round(time.clock()-t0,2))
        
    print("Time for MF :", round(time.clock()-t0,2), round(error(R,P,Q),3))

    return P, Q.T

ratings = np.loadtxt('../R.txt')#[]
ratings = ratings[0:100,0:100]
#for i in range(100):
#    for j in range(100):
#        ratings.append(30)
            
R = np.array(ratings).reshape(100,100)
P, Q = matrix_factorization(R)

