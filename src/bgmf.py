import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code = open('gpukernel.c', 'r').read()
mod = compiler.SourceModule(kernel_code)
matrixfact = mod.get_function("MatrixFactorization")

def matrix_factorization(UU,MM,RR, P, Q, ulimits, plimits, latent=30, gpu_steps=1, alpha=0.0002, beta=0.01, delta=0.01, debug=2):

    u_gpu = gpuarray.to_gpu(np.array(UU).astype(np.int32))
    v_gpu = gpuarray.to_gpu(np.array(MM).astype(np.int32))
    r_gpu = gpuarray.to_gpu(np.array(RR).astype(np.int32))

    a_gpu = gpuarray.to_gpu(np.array(P).astype(np.float32))
    b_gpu = gpuarray.to_gpu(np.array(Q).astype(np.float32))

    ul_gpu = gpuarray.to_gpu(np.array(ulimits).astype(np.int32))
    pl_gpu = gpuarray.to_gpu(np.array(plimits).astype(np.int32))

    t7 = time.clock()

    if debug>1:
        print("Length of uu,mm ", len(UU), len(MM), len(P), len(Q) )

    matrixfact(
        u_gpu, v_gpu, r_gpu, a_gpu, b_gpu,
        np.int32(latent), ul_gpu, pl_gpu, np.int32(gpu_steps),
        np.float32(alpha), np.float32(beta), np.float32(delta),
        block=(2,2,1),grid=(1,1)
    )

    P = a_gpu.get()
    Q = b_gpu.get()
    t8 = time.clock()

    if debug>1:
        print("Shape of P, Q :", P.shape, Q.shape)

    return P, Q
