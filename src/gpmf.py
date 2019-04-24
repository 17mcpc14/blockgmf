import numpy as np
import time
import math
from util import fetch, initUV
from error import rmse
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code = open('gpmf_kernel.c', 'r').read()

def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=30, steps=10, gpu_steps=2, alpha=0.0002, beta=0.02, delta=0.01, rmse_repeat_count=5, debug=1, dataset=''):

    U, V = initUV(int(np.max(users)+1), latent, int(np.max(movies)+1))
    U, V = np.array(U).astype(np.float32), np.array(V).astype(np.float32).transpose()

    print("Shape of P,Q : " , U.shape, V.shape)

    size = max(np.max(users), np.max(movies))
    split = int(size/blocks)

    mod = compiler.SourceModule(kernel_code)
    matrixfact = mod.get_function("MatrixFactorization")
    start_time=time.clock()
    y1, y2 = [], []
    error, count = 100, 0

    test_rmse = rmse(test_users, test_movies, test_ratings, U, V.T)
    print("Initial test error :", round(test_rmse,4))
    for k in range(steps):

        if debug>1:
            print("Step : ", k)

        t6 = time.clock()

        uu, mm, rr = np.array(users).astype(int), np.array(movies).astype(int), np.array(ratings).astype(int)

        t7 = time.clock()
        tools.clear_context_caches()
        u_gpu = gpuarray.to_gpu(uu)
        v_gpu = gpuarray.to_gpu(mm)
        r_gpu = gpuarray.to_gpu(rr)

        a_gpu = gpuarray.to_gpu(U)
        b_gpu = gpuarray.to_gpu(V)

        if debug>1:
            print("Length of uu,mm ", len(uu), len(mm), np.max(users), np.max(movies), U.shape, V.shape)

        if(len(uu)!=0 and len(mm)!=0):
            matrixfact(
                u_gpu, v_gpu, r_gpu, a_gpu, b_gpu,
                np.int32(np.max(users)), np.int32(latent), np.int32(np.max(movies)), np.int32(len(uu)), np.int32(len(mm)),np.int32(gpu_steps),
                np.float32(alpha), np.float32(beta), np.float32(delta),
                block=(16,16,1),grid=(3,4) # always keep blockIdx.z as 1 - the kernal expects no threads in z axis
            )
            P = a_gpu.get()
            Q = b_gpu.get()
            U,V = np.array(P),np.array(Q)
            t8 = time.clock()
            
            if debug>1:
                t9 = time.clock()
                if debug>2:
                    np.savetxt('U'+str(k),U,fmt='%.4f')
                    np.savetxt('V'+str(k),V,fmt='%.4f')
                print("Timer :", round(t7-t6,4), round(t8-t7,4), round(t9-t8,4))

        t5 = time.clock()
        if debug>1:
            print("Step time taken : ", round(t5-t7,2))
        y1.append(t5-start_time)
        test_rmse = rmse(test_users, test_movies, test_ratings, U, V.T)
        print("Step test error :", round(test_rmse,4))

        train_rmse = rmse(users, movies, ratings, U, V.T)
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
        np.savetxt('gpmf-'+str(blocks)+'-'+str(start_time)+'-y1.txt', y1, fmt='%.4f')
        np.savetxt('gpmf-'+str(blocks)+'-'+str(start_time)+'-y2.txt', y2, fmt='%.4f')
