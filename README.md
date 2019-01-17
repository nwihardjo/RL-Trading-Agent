# Reinforcement Learning Trading Agent

The repository contains implementation of an attempt to train RL agent to trade in the financial market using a *Deep Direct Reinforcement Learning* model through *Technical Analysis* methodon the data to provide more robust framework on top of the data using minute-based HKEX market data. Refer to the `UROP_1100_Report.pdf` for detailed explanation. 

The RL agent and its environment is defined in the `./model/` directory, the data is in `./data/` directory, and scripts for pre-processing the data is present in the `./data_wrangling/` directory.

Detailed documentation is provided on `README.md` in each folder and its corresponding files.

## Usage

To use and train the model, refer to the 3 notebooks where the model is trained on 3 different dataset. Refer to the `DRL_BOC+CNOOC.ipynb` for the most updated and refactored code with its documentation.

To use the your own data, replace and edit the data in the `./data/` directory.

## Dependencies

Using [`conda`] (https://anaconda.org/) package manager, build the environment with `environment.yml`:
```bash
$ conda env create -f environment.yml
```

Or without conda:
- Python 3.6
- TA-Lib >= 0.4.17. On some machines, underlying `TA-Lib` library should be installed in advance before installing using pip or other package manager. Refer [here](https://mrjbq7.github.io/ta-lib/install.html) for the predecessor library. 
- Pytorch >= 0.4.1
- tqdm >= 4.28.1
- pyfolio >= 0.9.0
- seaborn
- numpy
- pandas
- jupyter notebook
- and other basic packages.

## Installation

Clone the repo, and build virtual environment using `conda` for easier installation through the using the step above.

*Important note*: this project utilises `TA-Lib` library for python (Technical Analysis) as a part of the knowledge discovery on the data. It requires the predecessor `TA-Lib` library to be installed on your machine before the `TA-Lib` library for python can be installed properly (i.e. using `pip install talib` or conda). 

To install predecessor [`TA-Lib`](http://ta-lib.org/hdr_dw.html), refer [here](https://mrjbq7.github.io/ta-lib/install.html). 

## Contributor
- [Nathaniel Wihardjo] (https://github.com/nwihardjo/)
- [Hanif Dean] (https://github.com/hanifdean/)
- [Kenneth Lee] (https://github.com/kenneth-id/)

## Reference
- (https://github.com/yuriak/RLQuant/)
- [Recurrent Reinforcement Learning: A Hybrid Approach](https://arxiv.org/abs/1509.03044/)
- [Deep Direct Reinforcement Learning for Financial Signal Representation and Trading](https://ieeexplore.ieee.org/document/7407387/)
- [Using a Financial Training Criterion Rather than a Prediction Criterion ](https://www.worldscientific.com/doi/pdf/10.1142/S0129065797000422)
- [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
