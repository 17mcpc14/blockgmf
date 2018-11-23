# Block GPU Matrix Factorization

Matrix  factorization  is  well  known  technique  for  various machine  learning  problems  ranging  from  recommender  systems to text mining. The simplicity of the technique is based on deriving underlying latent factors leading to a behavior by utilization of large data of observed behaviors.

The  number  of  computations  for  matrix  factorization  of  adata  matrix X∈R{n×m} into  latent  feature  matrices U∈R{n×k},V∈R{k×m} is:≈(n×m×k×a + n×m×b)×N for some constants a and b, while N being number of iterations. Hence, the time complexity ≈O(n^4). As  n,  m → ∞, N and k remain  constants,  the  cost  of computation ≈n×m and the time complexity ≈O(n2). Asn,m →1^6, the number of computations≈10^12

Matrix  Factorization  on  a  GPU  gives  substantial  performance gain as the number of computations are nearly dividedby number of GPU cores leveraged. However, the amount of memory  available  on  a  typical  GPU  is  limited  and  for  large datasets it may not be possible to compute dense matrix factorization in one go with all data transferred into GPU cache. The memory required for a data matriX can be arrived as ≈(n×m + n×k + k×m)×c for some constant c bytes taken to represent each element. Hence the space complexity ≈O(n^2). Considering k as constant, and as n,m → 10^6, the numberof memory units required ≈10^12

The Block GPU matrix factorization considers the data matrix as a block matrix and factorization of each block is achieved on GPU. 

![alt text](https://github.com/17mcpc14/blockmf/misc/blockmf.png)

The approach is shown in the above figure considers  each  block  as  an  individual  matrix for  factorization. The algorithm factorizes each block for few  iterations  and  the  latent  features  of  each  of  the blocks are taken as  a  starting  point  for  computation  of  latent  features for  relevant  blocks  afterwards.  
![alt text](https://github.com/17mcpc14/blockmf/misc/example.png)

Above figure demonstrates a simple example with 4 blocks each factorized individually and then combined together to form U,V.


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


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
