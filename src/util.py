import numpy as np
import glob
from os.path import exists, basename
import sys
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
from sklearn.cluster import KMeans
#import pandas as pd
import time
import math
#import tables as tb
from scipy.sparse import csr_matrix
from scipy.io import loadmat

def fetch( u1, u2, v1, v2, users, movies, ratings):
    r = []
    u = []
    m = []
    
    for i in range(len(users)):
        if users[i] >=u1 and users[i] <= u2:
            if movies[i] >=v1 and movies[i] <= v2:
                #if i>=6040:
                #print(i, users[i], movies[i], ratings[i])
                u.append(users[i])
                m.append(movies[i])
                r.append(ratings[i])
            
    return np.array(u).astype(np.int32), np.array(m).astype(np.int32), np.array(r).astype(np.int32)

def initPQ(L,M,N):

    f = tb.open_file('./matrix-pt.h5', 'w')

    filters = tb.Filters(complevel=5, complib='blosc')
    ad = f.create_carray(f.root, 'a', tb.Float64Atom(), (L,M),
                        filters=filters)
    a = np.ones(M)
    for i in range(L):
        ad[i,:] = a

    bd = f.create_carray(f.root, 'b', tb.Float64Atom(), (M,N),
                        filters=filters)
    b = np.ones(N)
    for i in range(M):
        bd[i,:] = b

    return ad , bd
