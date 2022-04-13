# DCRNN-SQR: Diffusion Convolutional Recurrent Neural Network with Simultaneous Quantile Regression for Uncertainty Quantification in Traffic Forecasting

DCRNN-SQR is a diffusion convolutional recurrent neural network (DCRNN) with simultaneous quantile regression (SQR) loss for estimating both aleatoric (data) 
and epistemic (model) uncertainties. The SQR loss in DCRNN is used to predict the quantiles of the forecasting traffic distribution. A scalable Bayesian optimization 
based hyperparameter serach [DeepHyper](https://deephyper.readthedocs.io/en/latest/) is used to perform hyperparameter optimization on DCRNN-SQR, a selected set 
of high-performing configurations is used to fits a Gaussian copula model to capture the joint distributions of the hyperparameter configurations. Finally a set 
of high-performing configurations is sampled from the distribution and used to train an ensemble of DCRNN-SQR models.


## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* tables
* future
* mpi4py

## Download Data

The traffic data files for Los Angeles (METR-LA) are available at [METR-LA](https://anl.box.com/s/ptjgb2jcpf122jtooml5ew55x0ubibxq). The train, test, and validation data are available at `data/METR-LA/{train,val,test}.npz`. The adjacency matrix and configuration file are available at `METR-LA/sensor_graph/adj_mx.pkl` and `METR-LA/model/dcrnn_la.yaml`.

## Generate datasets 
Run the following comands to create the dataset with 100 hyperparameter configurations 

```no-highlight
mkdir data100
mv METR-LA data100
cd data100
mv METR-LA data00
for i in {1..99}; do cp -r data00 "data0$i"; done
python change_yaml.py
```

## Run the experiments 
We construct model ensemble using 100 synthetic hyperparameter configurations. We train 100 DCRNN-SQR models with synthetic hyperparameter configurations simultaneously on multiple GPUs.

To submit and run an experiment on the Cooley GPU cluster the following command is used:

```no-highlight
qsub qsub_100uq.sh
```

