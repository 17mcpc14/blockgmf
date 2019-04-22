#include <cstdio>
#include <stdlib.h>

__global__ void MatrixFactorization(int *u, int *v, int *r, float *p, float *q, int m, int *ulimits, int *plimits, int steps, float alpha, float beta, float delta)
{

   int L1 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y;//Lx*tx;
   int L2 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y+1;//Lx*tx;

   for(int s=0; s<steps; s++)
   {
       int c = 0;
       for (int i=ulimits[L1]; i < ulimits[L2]; i++)
       {
             int j = plimits[L1] + c;

             float eij = 0.0;
             for(int k=0; k<m; k++)
             {
                  eij += (p[j+k]*q[j+k]);
             }
             eij = r[i] - eij;

             for(int k=0; k<m; k++)
             {
                   if(isfinite(p[j+k] + alpha * (2 * eij * q[j+k] - beta * p[j+k])))
                   {
                        p[j +k] = p[j+k] + alpha * (2 * eij * q[j+k] - beta * p[j+k]);
                   }
                   if(isfinite(q[j+k] + alpha * (2 * eij * p[j+k] - beta * q[j+k])))
                   {
                        q[j+k] = q[j+k] + alpha * (2 * eij * p[j+k] - beta * q[j+k]);
                   }
             }
             c++;
        }
    } //steps
}
