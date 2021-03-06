import numpy as np
import tables as tb

def fetch( u1, u2, v1, v2, users, movies, ratings):
    r = []
    u = []
    m = []
    
    for i in range(len(users)):
        if users[i] >=u1 and users[i] <= u2:
            if movies[i] >=v1 and movies[i] <= v2:
                u.append(users[i])
                m.append(movies[i])
                r.append(ratings[i])
            
    return np.array(u).astype(np.int32), np.array(m).astype(np.int32), np.array(r).astype(np.int32)

def initUV(L,M,N):

    f = tb.open_file('./matrix-pt.h5', 'w')

    filters = tb.Filters(complevel=5, complib='blosc')
    ad = f.create_carray(f.root, 'a', tb.Float64Atom(), (L,M), filters=filters)
    a =  np.random.normal(0, .1, (M))*0.1
    for i in range(L):
        ad[i,:] = a

    bd = f.create_carray(f.root, 'b', tb.Float64Atom(), (N, M), filters=filters)
    for i in range(N):
        bd[i,:] = a

    return ad , bd
