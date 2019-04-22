import numpy as np
import time
import math
from util import fetch, initUV
import threading
from error import rmse
from scipy.sparse import csr_matrix

U,V = None, None

def block_factorization(P, Q, R, u1, u2, v1, v2, steps, K=30, alpha=0.0001, beta=0.01):
    
    t0 = time.clock()
    print("Block steps ",steps)
    if steps<1:
        steps = 1
  
    for step in range(steps):
        for i in range(u2-u1+1):
            for j in range(v2-v1+1):
                if R[i,j] > 0:
                    eij = R[i,j] - np.dot(P[i,:], Q[j, :])
                    for k in range(K):
                        if(np.isfinite(alpha * (2 * eij * Q[j,k] - beta * P[i,k]))):
                            P[i,k] = P[i,k] + alpha * (2 * eij * Q[j,k] - beta * P[i,k])
                        if(np.isfinite(alpha * (2 * eij * P[i,k] - beta * Q[j,k]))):
                            Q[j,k] = Q[j,k] + alpha * (2 * eij * P[i,k] - beta * Q[j,k])
        
    U[u1:u2+1, 0:K] = P.reshape( (u2-u1+1, K))
    V[v1:v2+1, 0:K] = Q.reshape( (v2-v1+1, K))
    
def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=30, steps=10, block_steps=1, alpha=0.00001, beta=0.01, delta=0.01, rmse_repeat_count=3, debug=2, dataset=''):
    global U, V
    U, V = initUV(np.max(users)+1, latent, np.max(movies)+1)
    R = csr_matrix((ratings, (users, movies))).todense()
    
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
            tpool = [None]*vs
            for j in range(vs):
                xtemp = int((i+stemp)%us)
                
                if debug>1:
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

                if debug>1:
                    print("Processing split : " , i , j, u1, u2, v1, v2)

                uu, mm, rr = fetch(u1, u2, v1, v2, users, movies, ratings)
                if debug>1:
                    print("Shapes of uu,mm,rr :", uu.shape, mm.shape, rr.shape)
                t6 = time.clock()
                P, Q = U[u1:u2+1, 0:latent], V[v1:v2+1, 0:latent]
                if debug>1:
                    print("P Q shapes : " , P.shape, Q.shape)
                t7 = time.clock()

                if debug>1:
                    print("Length of uu,mm ", len(uu), len(mm), u2-u1+1, v2-v1+1, P.shape, Q.shape)

                if(len(uu)!=0 and len(mm)!=0):
                    t = tpool[j]
                    if t is not None:
                        while t.isAlive():
                            print('waiting for the thread ...')
                            time.sleep(5)
                 	
                    t = threading.Thread(target=block_factorization, args=(P,Q,R, u1, u2, v1, v2, block_steps))
                    tpool[j] = t
                    t.start()
                    t8 = time.clock()

                stemp+=1

        t5 = time.clock()
        if debug>1:
            print(" Step time taken : ", round(t5-t4,2))
        y1.append(round(t5-start_time,3))
        test_rmse = rmse(test_users, test_movies, test_ratings, U, V)
        print("Step error :", round(test_rmse,3) )
        y2.append(round(test_rmse,3) )

        step_error=round(test_rmse,4)
        
        if step_error < delta:
            break
        elif error<step_error :
            break
        elif rmse_repeat_count<count:
            break
        elif error==step_error:
            count=count+1
        else:
            count = 0
        error=step_error

    np.savetxt(str(blocks*blocks)+'blocks_y2.txt', y2, fmt='%.3f')
    np.savetxt(str(blocks*blocks)+'blocks_y1.txt', y1, fmt='%.3f')    
