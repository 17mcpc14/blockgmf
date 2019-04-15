import numpy as np

def err(users, movies, ratings, P, Q):

    e = 0.0
    n = 0
    for idx in range(len(users)):
        u = users[idx]
        m = movies[idx]
        r = ratings[idx]

        if(r > 0):
            pred = np.dot(P[u,:], Q[m,:].T)
            diff = r - pred
            #print('Predicted :', pred, 'Actual :', r, 'diff:', diff, e, n, e*n/(n+1), diff/(n+1), diff*(diff/(n+1)) )
            e = e*n/(n+1) + diff*(diff/(n+1))
            n = n+1

    return np.sqrt(e)