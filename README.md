# banded-mm
**Disclaimer**: This repository was created as part of ETHZ Semester Project on "Efficient banded matrix multiplication for quantum transport simulation". I now use it as a personal reference for python projects, and additionally I use it for testing python dev tools. The stability of this package might not be guaranteed.

Primarily, **banded-mm** is a python package implementing a prototype of the BdGEMM algorithm developed during the ETHZ Semester Project. BdGEMM is BLAS-like algorithm for matrix multiplications involving two banded matrices. It allows for efficient block-wise multiplication of banded matrices on Nvidia GPU's. This solves memory constraints faced when trying to do multiplication of bigger matrices. Detailed reasoning about the algorithm can be found in the report "Efficient Banded Matrix Multiplication for Quantum Transport Simulation" in the `/docs` folder. 

`/src` contains `banded_mm` package which consists of two functions `BdMM` and `BdGEMM`. `BdGEMM` is available in multiple different implementations. If none is specified explicitly, it will use the most optimized implementation. The unit tests of `banded_mm` are located in `/tests`.

`/tools` contains utility python scripts with functions like `banded_matrix_generator` and scripts used for experimental evaluation of the algorithms.

## Install
**Requirements**
+ The latest tests were run on Windows 11 with WSL2 running Ubuntu LTE 22.04 and using a RTX3060. Alternative systems might work but are not guaranteed