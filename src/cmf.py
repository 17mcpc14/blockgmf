import numpy as np
import time
import math
from error import rmse

def matrix_factorization(users, movies, ratings, test_users, test_movies, test_ratings, K=10, steps=10, alpha=0.0002, beta=0.01, delta=0.01):

    t0 = time.clock()

    P, Q = np.ones((np.max(users)+1, K))*.2, np.ones((np.max(movies)+1, K))*.2  #initPQ(R.shape[0], K, R.shape[1])

    for step in range(steps):

        print("Step : ", step)

        for idx in range(len(users)):
            if(ratings[idx] > 0):
                i = users[idx]
                j = movies[idx]
                eij = ratings[idx] - np.dot(P[i,:], Q[j,:])

                for k in range(K):
                    P[i,k] = P[i,k] + alpha * (2 * eij * Q[j,k] - beta * P[i,k])
                    Q[j,k] = Q[j,k] + alpha * (2 * eij * P[i,k] - beta * Q[j,k])
        
        print("Time till now :", round(time.clock()-t0,2), "Train error", round(rmse(users, movies, ratings, P,Q), 4) , "Test error", round(rmse(test_users, test_movies, test_ratings, P,Q),4) )

    return P, Q
