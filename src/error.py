import numpy as np

def rmse(users, movies, ratings, P, Q):

    e = 0.0
    n = 0
    for idx in range(len(users)):
        u = users[idx]
        m = movies[idx]
        r = ratings[idx]

        try:
            if(r > 0):
                pred = np.dot(P[u,:], Q[m,:].T)
                diff = r - pred
                e = e*n/(n+1) + diff*(diff/(n+1))
                n = n+1
        except Exception, e:
            print("exception during error computation..", str(e), u, m, r)

    return np.sqrt(e)
