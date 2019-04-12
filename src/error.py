import numpy as np
import glob
from os.path import exists, basename
import sys
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
from sklearn.cluster import KMeans
#import pandas as pd
from pycuda import driver, compiler, gpuarray, tools
from skcuda import linalg
import pycuda.autoinit
import time
import math
#import tables as tb
from scipy.sparse import csr_matrix
from util import fetch, initPQ

kernel_rmse = """
#include <cstdio>
#include <stdlib.h>
__global__ void RMSE(float *p, float *q, int * u, int * v, int * r, float * ex, int * ey, int l, int m, int n, int u1, int u2, int v1, int v2, int ul, int vl)
{
   
      int tx = threadIdx.x + blockDim.x * blockIdx.x;
      int ty = threadIdx.y + blockDim.y * blockIdx.y;
 
      int Lx = 1 +( (l-1)/(blockDim.x * gridDim.x));
      int Ny = 1 + ( (n-1)/(blockDim.y * gridDim.y));
         
      int L1 =  Lx*tx;
      int L2 =  Lx*(tx+1);
      int N1 =  Ny*ty;
      int N2 =  Ny*(ty+1);
   
      int idx = tx*blockDim.x*gridDim.x +ty;
  
      for (int i=L1; i < L2; i++)
      {
            int rx = 0; 
            while(u[rx]< (u1+i) && rx <ul)
            {
                  rx++;
            }
            for(int j=N1; j<N2; j++)
            {
                  int rx0 = rx;
                  while(v[rx]< (v1+j) && rx <vl)
                  {
                        rx++;
                  }
                  if(u[rx]== (u1+i) && v[rx]== (v1+j) )
                  {
                     float Pvalue = 0;
               
                     for (int k = 0; k < m; k++) {
                        float Aelement = p[i*m +k];
                        float Belement = q[ j +k*n];
                        Pvalue += (Aelement * Belement);
                     }
                     Pvalue = round(Pvalue-r[rx]);
                     float temp = (ey[idx]+1);
                     ex[idx] = ((ex[idx]*(ey[idx]/temp)) + (Pvalue*(Pvalue/temp)));
                     ey[idx] = temp;
                  }
             }
      }
}
"""
modm = compiler.SourceModule(kernel_rmse)
rmse = modm.get_function("RMSE")

def mf_rmse(U, V, users, movies, ratings, split, latent=30, debug=1):
    
    us = int(math.ceil( np.float(np.max(users))/split ) )
    vs = int(math.ceil( np.float(np.max(movies))/split ) )
    
    u1, v1 = 0, 0
    error = 0.0
    totnum = 0
    totmse = 0.0
    t4 = time.clock()
    for i in range(us):
   
        u1 = i*split
        if np.max(users) < u1:
            u1 = int(np.max(users))

        u2 = ((i+1)*split - 1)
        if np.max(users) < u2:
            u2 = int(np.max(users))

        for j in range(vs):
            v1 = j*split
            if np.max(movies) < v1:
                v1 = int(np.max(movies))

            v2 = (j+1)*split -1
            if np.max(movies) < v2:
                v2 = int(np.max(movies))

            if debug>1:
                print("Processing split : " , i , j, u1, u2, v1, v2)
            
            uu, mm, rr = fetch(u1, u2, v1, v2, users, movies, ratings)
            if debug>1:
                print("Shapes of uu,mm,rr :", uu.shape, mm.shape, rr.shape)
           
            t6 = time.clock()
            P, Q = U[u1:u2+1, 0:latent], V[0:latent, v1:v2+1]
            P = P.reshape(P.shape[0]*P.shape[1], 1).astype(np.float32)
            Q = Q.reshape(Q.shape[0]*Q.shape[1], 1).astype(np.float32)

            tools.clear_context_caches()
            a_gpu = gpuarray.to_gpu(P)
            b_gpu = gpuarray.to_gpu(Q)

            t7 = time.clock()
            u_gpu = gpuarray.to_gpu(uu)
            v_gpu = gpuarray.to_gpu(mm)
            r_gpu = gpuarray.to_gpu(rr)

            ex_gpu = gpuarray.zeros((3072,1), np.float32)
            ey_gpu = gpuarray.zeros((3072,1), np.int32)
            
            if len(uu) > 0: 
                rmse(
                    a_gpu, b_gpu, u_gpu, v_gpu, r_gpu, ex_gpu, ey_gpu, 
                    np.int32(u2-u1+1), np.int32(latent), np.int32(v2-v1+1), np.int32(u1), np.int32(u2), np.int32(v1), np.int32(v2), np.int32(len(uu)), np.int32(len(mm)),
                    block=(16,16,1),
                    grid=(3,4)
                )
                ex = ex_gpu.get()
                ey = ey_gpu.get()
                num = np.sum(ey)
                mse = np.sum(np.dot(ex.T, ey))
                temp = np.float((totnum+num))
        
                error = error*(totnum/temp)+ (mse/temp)
                totnum += num
                totmse += mse
                if debug>1:
                    print(" mse , error ", totmse, mse, mse/num, error, num, len(uu))

            t8 = time.clock()


    return np.sqrt( error )
