import numpy as np
import glob
from os.path import exists, basename
import sys
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
import pandas as pd
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import time
import math
from scipy.sparse import csr_matrix
from util import initPQ, fetch
from error import mf_rmse

kernel_code = """

#include <cstdio>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatrixFactorization(int *u, int *v, int *r, float *p, float *q, float *ex, int * ey,float * eij, int l, int m, int n,int ul, int vl, int steps, float alpha, float beta, float delta)
{
   int tx = threadIdx.x + blockDim.x * blockIdx.x;
   int ty = threadIdx.y + blockDim.y * blockIdx.y;
   
   int Lx = 1 +( (l-1)/(blockDim.x * gridDim.x));
   int Ny = 1 + ( (n-1)/(blockDim.y * gridDim.y));
         
   int L1 = Lx*tx;
   int L2 = Lx*(tx+1);
   int N1 = Ny*ty;
   int N2 = Ny*(ty+1);
   
   if(L1 >= l)
   {
       return;
   }
   if(N1 >= n)
   {
       return;
   }
   
   if(L2 >= l)
   {
       L2=l;
   }
      
   if(N2 >= n)
   {
       N2=n;
   }
   
   for(int s=0; s<steps; s++)
   {

       for (int i=L1; i < L2; i++)
       {
	       for(int j=N1; j<N2; j++)
	       {
		    float predict = 0.0;
               
                    for (int k = 0; k < m; k++) {
                        float Aelement = p[i*m +k];
        	        float Belement = q[ j +k*n];
              	        predict += Aelement * Belement;
                    }

                    float eij = r[i*n+j] - predict;
		    
		    for(int k=0; k<m; k++)
		    {
                        if(isfinite(p[i*m +k] + alpha * (2 * eij * q[j +k*n] - beta * p[i*m +k])))
                        {
			    atomicAdd(&p[i*m +k] ,  alpha * (2 * eij * q[j +k*n] - beta * p[i*m +k]));
                        }
                        if(isfinite(q[j +k*n] + alpha * (2 * eij * p[i*m +k] - beta * q[j +k*n])))
                        {
			    atomicAdd(&q[j +k*n], alpha * (2 * eij * p[i*m +k] - beta * q[j +k*n]));
                        }
		    }
	       }
       }
      
   } // steps

}    
"""

def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=30, steps=10, gpu_steps=2, alpha=0.000001, beta=0.01, delta=0.01, rmse_repeat_count=10, debug=2, dataset=''):
        
    U, V = initPQ(int(np.max(users)+1), latent, int(np.max(movies))+1)
    size = max(np.max(users)+1, np.max(movies)+1)
    split = int(size/blocks)
    us = int(math.ceil( np.float(np.max(users))/split ) )
    vs = int(math.ceil( np.float(np.max(movies))/split ) )
    if debug>1:
        print("Total splits : ",split, us, vs, us*vs)

    mod = compiler.SourceModule(kernel_code)
    matrixfact = mod.get_function("MatrixFactorization")
    start_time=time.clock()
    y1, y2 = [], []
    flag1, count = 1000, 0
    totnum = 0
    totmse = 0.0

    for k in range(steps):

        if debug>1: 
            print("Step : ", k)
    
        error = 0
        u1, v1 = 0, 0
          
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
                #if debug>1:
                #    print("Shapes of uu,mm,rr :", uu.shape, mm.shape, rr.shape)
            
                t6 = time.clock()
                P, Q = U[u1:u2+1, 0:latent], V[0:latent, v1:v2+1]
                P = P.reshape(P.shape[0]*P.shape[1], 1).astype(np.float32)
                Q = Q.reshape(Q.shape[0]*Q.shape[1], 1).astype(np.float32)
    
                t7 = time.clock()
                tools.clear_context_caches()
                u_gpu = gpuarray.to_gpu(uu)
                v_gpu = gpuarray.to_gpu(mm)
                r_gpu = gpuarray.to_gpu(rr)
    
                a_gpu = gpuarray.to_gpu(P) 
                b_gpu = gpuarray.to_gpu(Q)
        
                ex_gpu = gpuarray.zeros((3072,1), np.float32)
                ey_gpu = gpuarray.zeros((3072,1), np.int32)

                eij_gpu = gpuarray.empty(((v2-v1+1)*(u2-u1+1),1), np.float32)
       
                #if debug>1:
                #    print("Length of uu,mm ", len(uu), len(mm), u2-u1+1, v2-v1+1, P.shape, Q.shape)
                
                if(len(uu)!=0 and len(mm)!=0): 
                    matrixfact(
                        u_gpu, v_gpu, r_gpu, a_gpu, b_gpu, ex_gpu, ey_gpu, eij_gpu, 
                        np.int32(u2-u1+1), np.int32(latent), np.int32(v2-v1+1), np.int32(len(uu)), np.int32(len(mm)),np.int32(gpu_steps),
                        np.float32(alpha), np.float32(beta), np.float32(delta),
                        block=(16,16,1),grid=(3,4)
                    )
                    P = a_gpu.get()
                    Q = b_gpu.get()
                    t8 = time.clock()               
                
                    #if debug>1:
                    #    print("Shape of P, Q :", P.shape, Q.shape)

                    U[u1:u2+1, 0:latent] = P.reshape( (u2-u1+1, latent))
                    V[0:latent, v1:v2+1] = Q.reshape( (latent, v2-v1+1))
                    t9 = time.clock()               
                    if debug>2:
                        np.savetxt('U'+str(k),U,fmt='%.4f')
                        np.savetxt('V'+str(k),V,fmt='%.4f')     
                    #if debug>1:
                    #    print("Timer :", t7-t6, t8-t7, t9-t8)
                    
                    ex = ex_gpu.get()
                    ey = ey_gpu.get()
                    num = np.sum(ey)

                    #print("Count check :", num, len(uu))

                    mse = np.sum(np.dot(ex.T, ey))
                    temp = np.float((totnum+num))

                    error = error*(totnum/temp)+ (mse/temp)
                    totnum += num
                    totmse += mse
                    if debug>1:
                        print("Completed processing : ", i , j, round(np.sqrt(error),3), len(uu))
        
        t5 = time.clock()
        test_rmse = mf_rmse(U, V , test_users, test_movies, test_ratings, min(split, max(np.max(test_users)+1, np.max(test_movies)+1)), latent=latent, debug=0)
        if debug>1:
            print(" Step time taken, error : ", round(t5-t4,3), round(np.sqrt(error),3), round(test_rmse,3))
        y1.append(t5-start_time)
       
        train_rmse = error #mf_rmse(U, V , users, movies, ratings, split, latent=latent, debug=debug)
        y2.append([train_rmse,test_rmse])
    
        flag=round(test_rmse,3)

        if flag < delta:
            print("flag less than delta")
            break
        elif flag>flag1 :
            print("flag great than flag1")
            break
        elif rmse_repeat_count==count:
            print("count == 10")
            break
        elif flag==flag1:
            count=count+1
        else:
            count=0
        flag1=flag
            
    if debug>1:
        np.savetxt('../log/bgmf-'+dataset+'-'+str(blocks)+'-'+str(start_time)+'.txt', y1, fmt='%.4f')
        np.savetxt('../log/bgmf-'+dataset+'-'+str(blocks)+'-'+str(start_time)+'.txt', y2, fmt='%.4f')
