# Block based approaches to Matrix Factorization

Matrix  factorization (MF) is  well  known  technique in machine  learning. MF is used to solve problems  ranging  from  recommender  systems, dimensionality reduction, text mining, astronomy, bio informatics, medical image processing etc. The simplicity of the technique is based on deriving underlying latent factors leading to a behavior by using large volume of observed behaviors.

MF on GPUs  gives  substantial  performance gain as the computation can be parallelized on to each GPU core. However, the amount of memory  available  on  a  typical  GPU  is  limited  and  for  large datasets it may not be possible to compute  MF in one go with all data transferred into GPU cache.

Block based approaches to MF is an attempt to achieve parallelism, scalability, GPU computation and distributed computation through a single framework. 

The Block based approaches to matrix factorization (BMF) considers the data matrix as a block matrix and factorization is achieved at block level. Below figure demonstrates a simple example with 4 block matrix, with each block factorized individually and then combined together to form U,V. 

<img src="misc/example.png" width="350" title="Example of BMF">

## Getting Started


### Prerequisites

```
Python 3.0
CUDA v8.0
pyCUDA v1.8

The GPU based Block matrix factorization has been developed using python using pyCUDA support.

```

### Installing


```
git clone 'https://github.com/17mcpc14/blockgmf'

```

## Running the tests

The project contains multiple variants (implementations) of BMF:
```
1. src/cmf.py - CPU based BMF implementation without parallelism
2. src/cpmf.py - Multi-threaded CPU based BMF implementation
3. src/bgmf.py - Block based GPU implementation of BMF
4. src/bcsvd.py - CPU based implementation of BMF with SVD kernel
5. src/bgsvd.py - GPU based implementation of BMF with SVD kernel

All above programs can be invoked through invocation of respective MF methods (factorize/block_factorization). Alternatively the same can be invoked from respective main programs as below:

1. src/main_cmf.py <ml-100k/train.csv> <ml-100k/test.csv> <2000>
2. src/main_cpmf.py <ml-100k/train.csv> <ml-100k/test.csv> <2000> <16>
3. src/main_bgmf.py <ml-100k/train.csv> <ml-100k/test.csv> <16> <2000> <1>
4. src/main_bcsvd.py <ml-100k/train.csv> <ml-100k/test.csv> <16> <2000> <1>
5. src/main_bgsvd.py <ml-100k/train.csv> <ml-100k/test.csv> <16> <2000> <1>

```

**Note:** the programs can be run with Rtrain.csv, Rtest.csv uploaded into /data directory of the project or with any other dataset by modifying the main_<algo>.py accordingly. 

## Authors

* **Prasad Bhavana** - *Initial work* 

## Acknowledgments

* Vineet C Padmanabhan, Professor, University of Hyderabad
