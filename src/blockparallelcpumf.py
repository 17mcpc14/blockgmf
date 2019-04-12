import numpy as np
import glob
from os.path import exists, basename
import sys
import time
import math
import numpy as np
from util import fetch
import threading

args=list(sys.argv)
test_input =args[2]
train_input =args[1]
blocks=int(args[3])
steps = int(args[4])
gpu_steps = int(args[5])

def error(R, P , Q, u1, u2, v1, v2):
    Rc = np.dot(P,Q)
    Rc = Rc - R[u1:u2+1,v1:v2+1]
    Rc = np.square(Rc)
    error = np.sum(Rc)/(P.shape[0]*Q.shape[1])
    return np.sqrt(error)

def block_factorization(P, Q, R, u1, u2, v1, v2, steps, K=10, alpha=0.0001, beta=0.01):
    
    t0 = time.clock()
    steps = int(steps/1)
    print("Steps ",steps)
    if steps<1:
        steps = 1
  
    Rc = np.zeros((R.shape[0], R.shape[1])) #initPQ(R.shape[0], K, R.shape[1])

    x, y1, y2 = [], [], []
    
    flag1 = 1000.0
    count = 0
    for step in range(steps):
        #print("Step : " , step)
        
        for i in range(u2-u1+1):
            for j in range(v2-v1+1):
                eij = R[i,j] - np.dot(P[i,:], Q[:,j])
                for k in range(K):
                    if(np.isfinite(alpha * (2 * eij * Q[k,j] - beta * P[i,k]))):
                        P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
                    if(np.isfinite(alpha * (2 * eij * P[i,k] - beta * Q[k,j]))):
                        Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
        
    U[u1:u2+1, 0:latent] = P.reshape( (u2-u1+1, latent))
    V[0:latent, v1:v2+1] = Q.reshape( (latent, v2-v1+1))

def factorize(users, movies, ratings, test_users, test_movies, test_ratings, blocks=1, latent=10, steps=10, gpu_steps=2, alpha=0.00001, beta=0.01, delta=0.01, rmse_repeat_count=3, debug=2, dataset=''):

    #U, V = np.ones((R.shape[0], latent)), np.ones((latent, R.shape[1]))
    size = max(np.max(users)+1, np.max(movies)+1)
    split = int(size/blocks)
    us = int(math.ceil( np.float(np.max(users))/split ) )
    vs = int(math.ceil( np.float(np.max(movies))/split ) )
    if debug>1:
        print("Total splits : ",split, us, vs, us*vs)
        print("U, V shapes :", U.shape, V.shape)

    start_time=time.clock()
    y1, y2 = [], []
    count = 0

    flag1 = error(R, U, V,0, R.shape[1], 0, R.shape[0]) 
    
    for k in range(steps):

        if debug>1:
            print("Step : ", k)

        rmse = 0
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

                #print("Processing split : " , i , j, u1, u2, v1, v2)

                uu, mm, rr = fetch(u1, u2, v1, v2, users, movies, ratings)
                if debug>1:
                    print("Shapes of uu,mm,rr :", uu.shape, mm.shape, rr.shape)
                t6 = time.clock()
                P, Q = U[u1:u2+1, 0:latent], V[0:latent, v1:v2+1]
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
                 	
                    t = threading.Thread(target=block_factorization, args=(P,Q,R, u1, u2, v1, v2, gpu_steps))
                    tpool[j] = t
                    t.start()
                    t8 = time.clock()

                stemp+=1

        t5 = time.clock()
        if debug>1:
            print(" Step time taken : ", round(t5-t4,2))
        y1.append(round(t5-start_time,3))
        test_rmse = error(R, U, V,0, R.shape[1], 0, R.shape[0]) #e(U, V , test_users, test_movies, test_ratings, min(split, max(np.max(test_users), np.max(test_movies))), latent=latent, debug=debug)
        print("Step error :", round(test_rmse,3) )
        y2.append(round(test_rmse,3) )

        flag=round(test_rmse,4)
        gpu_steps = int(gpu_steps*flag/flag1)

        #if flag < delta:
        #    break
        #elif flag1<flag :
        #    break
        #elif rmse_repeat_count<count:
        #    break
        #elif flag==flag1:
        #    count=count+1
        #else:
        #    count = 0
        #flag1=flag

    np.savetxt(str(blocks*blocks)+'blocks_'+str(gpu_steps)+'iterations_y2.txt', y2, fmt='%.3f')
    np.savetxt(str(blocks*blocks)+'blocks_'+str(gpu_steps)+'iterations_y1.txt', y1, fmt='%.3f')

R = np.loadtxt('../R.txt')
users = []
movies = []
ratings = []
for i in range(1000):
    for j in range(1000):
        users.append(i)
        movies.append(j)
        ratings.append(R[i,j])

latent = 10
U, V = np.ones((R.shape[0], latent)), np.ones((latent, R.shape[1]))            
factorize(users, movies, ratings, users, movies, ratings,blocks=blocks, steps=steps, gpu_steps=gpu_steps, debug=1)


