#include <cstdio>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatrixFactorization(int *u, int *v, int *r, float *p, float *q, int l, int m, int n,int ul, int vl, int steps, float alpha, float beta, float delta)
{
   int tx = threadIdx.x + blockDim.x * blockIdx.x;
   int ty = threadIdx.y + blockDim.y * blockIdx.y;
   
   int Lx = 1 +( (l-1)/(blockDim.x * gridDim.x));
   int Ny = 1 + ( (n-1)/(blockDim.y * gridDim.y));
         
   int L1 = Lx*tx;
   int L2 = Lx*(tx+1);
   int N1 = Ny*ty;
   int N2 = Ny*(ty+1);
   
   if(L1 > l)
   {
       return;
   }
   if(N1 > n)
   {
       return;
   }
   
   if(L2 > l)
   {
       L2=l+1;
   }
   if(N2 > n)
   {
       N2=n+1;
   }
   
   for(int s=0; s<steps; s++)
   {
       for (int i=L1; i < L2; i++)
       {

               for(int j=N1; j<N2; j++)
               {
                    int rx = 0;
                    while(u[rx]!=i && rx <ul)
                    {
                        rx++;
                    }
                    while(v[rx]!=j && rx <vl)
                    {
                        rx++;
                    }
                    int rating = 0;
                    if(u[rx]==i && v[rx]==j)
                    {
                        rating = r[rx];
                    }
                    if(rating == 0)
                    {
                        continue;
                    }

                    float Pvalue = 0.0;
                    for (int k = 0; k < m; k++) {
                        float Aelement = p[i*m +k];
                        float Belement = q[ j +k*n];
                        Pvalue += Aelement * Belement;
                    }

                    float eij = rating - Pvalue;
                    if(isfinite(eij)){
                        for(int k=0; k<m; k++)
                        {
                            if(isfinite(p[i*m +k] + alpha * (2* eij * q[j +k*n] - beta * p[i*m +k])))
                            {
                                atomicAdd(&p[i*m +k] , alpha * ( 2 * eij * q[j +k*n] - beta * p[i*m +k]));
                            }
                            if(isfinite(q[j +k*n] + alpha * (2 * eij * p[i*m +k] - beta * q[j +k*n])))
                            {
                                atomicAdd(&q[j +k*n] , alpha * ( 2 * eij * p[i*m +k] - beta * q[j +k*n]));
                            }
                       }
                    }
               }
       }

   } // steps

}
}    

                             
