import numpy as np
import time
import math
from util import fetch, initUV
from error import rmse
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code = open('gpmfkernel.c', 'r').read()
mod = compiler.SourceModule(kernel_code)
matrixfact = mod.get_function("MatrixFactorization")

def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=30, steps=10, gpu_steps=2, alpha=0.000001, beta=0.01, delta=0.01, rmse_repeat_count=5, debug=1, dataset=''):

    U, V = initUV(int(np.max(users)+1), latent, int(np.max(movies))+1)
    size = max(np.max(users), np.max(movies))
    split = int(size/blocks)
    us = int(math.ceil( np.float(np.max(users))/split ) )
    vs = int(math.ceil( np.float(np.max(movies))/split ) )
    if debug>1:
        print("Total splits : ",split, us, vs, us*vs)

    mod = compiler.SourceModule(kernel_code)
    matrixfact = mod.get_function("MatrixFactorization")
    start_time=time.clock()
    y1, y2 = [], []
    error, count = 100, 0

    for k in range(steps):

        if debug>1:
            print("Step : ", k)

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
                np.max(movies) < v2:
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

                t7 = time.clock()
                tools.clear_context_caches()
                u_gpu = gpuarray.to_gpu(uu)
                v_gpu = gpuarray.to_gpu(mm)
                r_gpu = gpuarray.to_gpu(rr)

                a_gpu = gpuarray.to_gpu(P)
                b_gpu = gpuarray.to_gpu(Q)

                e_gpu = gpuarray.empty((3072,1), np.float32)
                eij_gpu = gpuarray.empty(((v2-v1+1)*(u2-u1+1),1), np.float32)

                if debug>1:
                    print("Length of uu,mm ", len(uu), len(mm), u2-u1+1, v2-v1+1, P.shape, Q.shape)

                if(len(uu)!=0 and len(mm)!=0):
                    matrixfact(
                        u_gpu, v_gpu, r_gpu, a_gpu, b_gpu, e_gpu, eij_gpu,
                        np.int32(u2-u1+1), np.int32(latent), np.int32(v2-v1+1), np.int32(len(uu)), np.int32(len(mm)),np.int32(gpu_steps),
                        np.float32(alpha), np.float32(beta), np.float32(delta),
                        block=(16,16,1),grid=(3,4)
                    )
                    P = a_gpu.get()
                    Q = b_gpu.get()
                    t8 = time.clock()

                    if debug>1:
                        print("Shape of P, Q :", P.shape, Q.shape)

                    U[u1:u2+1, 0:latent] = P.reshape( (u2-u1+1, latent))
                    V[0:latent, v1:v2+1] = Q.reshape( (latent, v2-v1+1))
                    t9 = time.clock()
                    if debug>2:
                        np.savetxt('U'+str(k),U,fmt='%.4f')
                        np.savetxt('V'+str(k),V,fmt='%.4f')
                    if debug>1:
                        print("Timer :", t7-t6, t8-t7, t9-t8)
                        print("Completed processing : ", i , j)

        t5 = time.clock()
        if debug>1:
            print(" Step time taken : ", round(t5-t4,2))
        y1.append(t5-start_time)
        test_rmse = rmse(test_users, test_movies, test_ratings, U, V)
        train_rmse = rmse(users, movies, ratings, U, V)
        y2.append([train_rmse,test_rmse])

        step_error=round(test_rmse,4)

        if step_error < delta:
            break
        elif step_error==error:
            count=count+1
        elif step_error>error:
            break
        elif rmse_repeat_count==count:
            break
        else:
            error=step_error

    if debug>1:
        np.savetxt('../log/gpmf-'+str(blocks)+'-'+str(start_time)+'.txt', y1, fmt='%.4f')
        np.savetxt('../log/gpmf-'+str(blocks)+'-'+str(start_time)+'.txt', y2, fmt='%.4f')
