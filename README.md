# Block GPU Matrix Factorization

Matrix  factorization  is  well  known  technique  for  variousmachine  learning  problems  ranging  from  recommender  sys-tems to text mining. The simplicity of the technique is basedon deriving underlying latent factors leading to a behavior byutilization of large data of observed behaviors

The  number  of  computations  for  matrix  factorization  of  adata  matrix X∈R{n×m} into  latent  feature  matrices U∈R{n×k},V∈R{k×m}can be arrived as:≈(n×m×k×a+n×m×b)×Nfor some constants a and b, whileNnumber of iterations.Hence the time complexity≈O(n4)As  n,  m→ ∞,Nandkremain  constants,  the  cost  ofcomputation≈n×mand the time complexity≈O(n2).Asn,m→106, the number of computations≈1012

Matrix  Factorization  on  a  GPU  gives  substantial  perfor-mance gain as the number of computations are nearly dividedby number of GPU cores leveraged. However, the amount ofmemory  available  on  a  typical  GPU  is  limited  and  for  largedatasets it may not be possible to compute matrix factorizationat once with all data transferred into GPU cache.

Consideringkas constant, and asn,m→10^6, the numberof memory units required≈10^12

Introduced    below    is    a    novel    approach    that    considersany  given  matrix  into  a  block  matrix  with  each  sub  matrixindependently    converged    within    a    GPU,    the    resultantlatent  feature  sub-matrices  reused  for  factorization  of  otherrelevant  sub-matrices  and  the  overall  process  is  repeated  tillconvergence equivalent to regular matrix factorization.
## Getting Started


### Prerequisites

Python 3.0
CUDA v8.0
pyCUDA v1.8
```
The GPU based Block matrix factorization has been developed using python using pyCUDA support.

```

### Installing

git clone 'https://github.com/17mcpc14/blockgmf'

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

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
