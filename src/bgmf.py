import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, coo_matrix
import time
import math
from error import rmse
from util import fetch, initUV
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code = open('bgmf_kernel.c', 'r').read()
mod = compiler.SourceModule(kernel_code)
matrixfact = mod.get_function("MatrixFactorization")

def matrix_factorization(UU,MM,RR, U, V, ulimits, umin, mmin, latent=30, gpu_steps=1, alpha=0.0002, beta=0.01, delta=0.01, debug=2):

    u_gpu = gpuarray.to_gpu(np.array(UU).astype(np.int32))
    v_gpu = gpuarray.to_gpu(np.array(MM).astype(np.int32))
    r_gpu = gpuarray.to_gpu(np.array(RR).astype(np.int32))

    a_gpu = gpuarray.to_gpu(np.array(U).astype(np.float32))
    b_gpu = gpuarray.to_gpu(np.array(V).astype(np.float32))

    ul_gpu = gpuarray.to_gpu(np.array(ulimits).astype(np.int32))

    t7 = time.clock()

    if debug>1:
        print("Length of uu,mm ", len(UU), len(MM), len(U), len(V) )

    matrixfact(
        u_gpu, v_gpu, r_gpu, a_gpu, b_gpu,
        np.int32(latent), ul_gpu, np.int32(umin), np.int32(mmin), np.int32(gpu_steps),
        np.float32(alpha), np.float32(beta), np.float32(delta),
        block=(16,16,1),grid=(3,2)
    )

    P = a_gpu.get()
    Q = b_gpu.get()
    t8 = time.clock()

    if debug>1:
        print("Shape of P, Q :", P.shape, Q.shape)

    return P, Q

def pack(UU, MM, RR, uu, mm, rr, ulimits):

    ulimits.append(len(uu))

    UU.extend(uu)
    MM.extend(mm)
    RR.extend(rr)

    return UU, MM, RR, ulimits

def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=10, steps=10, gpu_steps=2, alpha=0.0002, beta=0.01, delta=0.01, rmse_repeat_count=3, debug=2, dataset=''):

    U, V = initUV( np.max(users)-np.min(users)+1, latent, np.max(movies)-np.min(movies)+1)
    U = np.array(U)
    V = np.array(V)

    size = max(np.max(users)+1, np.max(movies)+1)
    split = int(size/blocks)
    us = int(math.ceil( np.float(np.max(users))/split ) )
    vs = int(math.ceil( np.float(np.max(movies))/split ) )
    if debug>1:
        print("Total splits : ",split, us, vs, us*vs)
        print("U, V shapes :", U.shape, V.shape)

    start_time=time.clock()
    y1, y2 = [], []
    count, error = 0, 100
    
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

            stemp = 0
            UU, MM, RR = [], [], []
            ulimits = [0]
           
            for j in range(vs):
                xtemp = int((i+stemp)%us)

                print("i, j, ii, jj ", i, j, xtemp, j)

                u1 = xtemp*split
                if np.max(users) < u1:
                    u1 = int(np.max(users))

                u2 = ((xtemp+1)*split - 1)
                if np.max(users) < u2:
                    u2 = int(np.max(users))

                v1 = j*split
                if np.max(movies) < v1:
                    v1 = int(np.max(movies))
                    
                v2 = (j+1)*split -1
                if np.max(movies) < v2:
                    v2 = int(np.max(movies))

                print("Processing split : " , i , j, u1, u2, v1, v2)

                uu, mm, rr = fetch(u1,u2, v1,v2, users,movies,ratings)

                if(len(uu)!=0 and len(mm)!=0):
                    UU,MM,RR, ulimits = pack(UU,MM,RR, uu,mm,rr, ulimits)

                stemp+=1
            U, V = matrix_factorization(UU,MM,RR, U,V, ulimits,np.min(users), np.min(movies))

        t5 = time.clock()
        if debug>1:
            print(" Step time taken : ", round(t5-t4,2))

        y1.append(round(t5-start_time,3))
        train_rmse = rmse(users, movies, ratings, U, V)
        test_rmse = rmse(test_users, test_movies, test_ratings, U, V)
        print("Train error:", round(train_rmse, 3) , " Test error:", round(test_rmse,3) )
        y2.append(round(test_rmse,3) )

        step_error=round(test_rmse,4)
        
        if step_error < delta:
            break
        elif error<step_error :
            break
        elif rmse_repeat_count<count:
            break
        elif step_error==error:
            count=count+1
        else:
            count = 0
        error=step_error

    np.savetxt('blocks_'+str(gpu_steps)+'iterations_y2.txt', y2, fmt='%.3f')
    np.savetxt('blocks_'+str(gpu_steps)+'iterations_y1.txt', y1, fmt='%.3f')
