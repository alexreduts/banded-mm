# banded-mm
ETHZ Semester Project on "Efficient banded matrix multiplication for quantum transport simulation"

Library with prototype implementations of BdGEMM Algorithm and comparison Algorithms BdMM, etc.

Detailed reasoning about the algorithm can be found in the in the Report pdf in the docs section

**Note**: This repository also serves as a personal reference for python projects, and is used for
testing dev tools.

Library Design: the banded matrix multiplication offers two functions when imported
BdMM and BdGEMM, The library offers different implementations variations of BdGEMM which can be
specified with a argument `blocking` etc. If not specified it will default to naiveCopy


## Install
**Requirements**
+ Tested on Ubuntu LTE 22.04 and Quatro P2000 alternative systems might work but are not guaranteed
+ CUDA version ... and compatible Graphics Card from Nvidia

1. Clone Repo `git clone ...`
2. Install poetry `link to poetry`
3. cd ... && pip install --editable .

## Development Notes
### Testing
unittest
tox
(pytest)

precommit
    flake8
    black
    _test_



## Run Experiments
+ Create python script running with different parameters given in a TOML config file