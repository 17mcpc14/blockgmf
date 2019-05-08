#include <cstdio>
#include <stdlib.h>

__global__ void MatrixFactorization(int *u, int *v, int *r, float *p, float *q, int m, int *ulimits, int umin, int vmin, int steps, float alpha, float beta, float delta)
{

   int L1 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y;//Lx*tx;
   int L2 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y+1;//Lx*tx;

   for(int s=0; s<steps; s++)
   {
       int i1 = ulimits[L1], i2 = ulimits[L2] ;
       for (int i=i1; i < i2; i++)
       {
             int pidx = u[i]-umin;
             int qidx = v[i]-vmin;

             float eij = 0.0;
             for(int k=0; k<m; k++)
             {
                  eij += (p[pidx*m+k]*q[qidx*m+k]);
             }
             eij = r[i] - eij;

             for(int k=0; k<m; k++)
             {
                   if(isfinite(p[pidx*m+k] + alpha * (2 * eij * q[qidx*m+k] - beta * p[pidx*m+k])))
                   {
                        atomicAdd(&p[pidx*m+k] , alpha * (2 * eij * q[qidx*m+k] - beta * p[pidx*m+k]));
                   }
                   if(isfinite(q[qidx*m+k] + alpha * (2 * eij * p[pidx*m+k] - beta * q[qidx*m+k])))
                   {
                        atomicAdd(&q[qidx*m+k] , alpha * (2 * eij * p[pidx*m+k] - beta * q[qidx*m+k]));
                   }
             }
        }
    } //steps
}
