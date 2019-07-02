#include <cstdio>
#include <stdlib.h>

__global__ void SGD(int *u, int *v, int *r, float *p, float *q, int m, int *ulimits, int *bu, int *bi, int gmean, int umin, int vmin, int steps, float alpha, float beta, float delta)
{

   int L1 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y;//Lx*tx;
   int L2 = (blockDim.x*blockIdx.x)*(blockIdx.y*blockDim.y)+(blockDim.x*threadIdx.x)+threadIdx.y+1;//Lx*tx;
   float alphau = 0.019;
   float alpham = 0.004;
   float betau = 0.019;
   float betam = 0.019;
   float alphabu = 0.004;
   float alphabi = 0.013;
   float betabu = 0.019;
   float betabi = 0.007;

   for(int s=0; s<steps; s++)
   {
       int i1 = ulimits[L1], i2 = ulimits[L2] ;
       //if (i2>0)
       //    printf("U1 %d U2 %d %d %d %d %d %d %d \n", i1, i2, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
       for (int i=i1; i < i2; i++)
       {
             int pidx = u[i]-umin;
             int qidx = v[i]-vmin;
             //printf("u %d , v %d \n", pidx, qidx);

             float eij = 0.0;
             for(int k=0; k<m; k++)
             {
                  eij += (p[pidx*m+k]*q[qidx*m+k]);
             }
             eij = r[i] - gmean - bu[pidx] - bi[qidx] - eij;
           
             bu[pidx] = alphabu * (eij - betabu*bu[pidx]);
             bi[qidx] = alphabi * (eij - betabi*bi[qidx]);

             for(int k=0; k<m; k++)
             {
 
                   if(isfinite(p[pidx*m+k] + alphau * (eij * q[qidx*m+k] - betau * p[pidx*m+k])))
                   {
                        atomicAdd(&p[pidx*m+k] , alphau * (eij * q[qidx*m+k] - betau * p[pidx*m+k]));
                   }
                   if(isfinite(q[qidx*m+k] + alpham * (eij * p[pidx*m+k] - betam * q[qidx*m+k])))
                   {
                        atomicAdd(&q[qidx*m+k] , alpham * (eij * p[pidx*m+k] - betam * q[qidx*m+k]));
                   }
             }
        }
    } //steps
}
