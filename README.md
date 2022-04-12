# DCRNN-SQR: Diffusion Convolutional Recurrent Neural Network with Simultaneous Quantile Regression for Uncertainty Quantification in Traffic Forecasting

DCRNN-SQR is a diffusion convolutional recurrent neural network (DCRNN) with simultaneous quantile regression (SQR) loss for estimating both aleatoric (data) 
and epistemic (model) uncertainties. The SQR loss in DCRNN is used to predict the quantiles of the forecasting traffic distribution. A scalable Bayesian optimization 
based hyperparameter serach [DeepHyper](https://deephyper.readthedocs.io/en/latest/) is used to perform hyperparameter optimization on DCRNN-SQR, a selected set 
of high-performing configurations is used to fits a Gaussian copula model to capture the joint distributions of the hyperparameter configurations. Finally a set 
of high-performing configurations is sampled from the distribution and used to train an ensemble of DCRNN-SQR models.


