import os
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf      
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from scipy import stats
from math import sqrt
import mc_dropout

# Global variable
hidden_dim = [50,50,50]

# Main Function
def test_mc_dropout(X_train, y_train,
                   X_val, y_val,
                   X_test, y_test,
                   alpha):
    
    # Parameter Grid for search
    tau_grid= np.logspace(-3,0,5)
    dropout_grid= [.2,.3,.4,.5,.6,.7]

    # Initialize hyperparmeters
    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0

    # Cross- Validation
    for dropout_rate in dropout_grid:
        for tau in tau_grid:
            # Train network
            network = mc_dropout.net(X_train, y_train, n_hidden= hidden_dim, normalize = True, 
                              n_epochs = 10, tau = tau, dropout = dropout_rate)

            # Validate hyperparameters on log likelihood
            _, ll = network.predict(X_val, y_val)

            if (sum(ll.squeeze()) > best_ll):
                best_ll = sum(ll.squeeze())
                best_network = network
                best_tau = tau
                best_dropout = dropout_rate

        # Train optimum network
        best_network= mc_dropout.net(X_train, y_train, n_hidden= hidden_dim, normalize = True, 
                                  n_epochs = 50, tau = best_tau, dropout = best_dropout)


        # Predict 1. Monte Carlo samples, 2. log-likelihood for each datapoint from the T samples
        Yt_hat, ll= best_network.predict(X_test, y_test)
        T = Yt_hat.shape[0]
        
        # Mean prediction
        MC_pred = np.mean(Yt_hat, 0)
        
        # Variance for each prediction
        Var = np.ndarray(shape=(Yt_hat.shape[1]))
        for datapoint in range(Yt_hat.shape[1]):
            Eyy= 1/T * Yt_hat[:,datapoint,0].transpose() @ Yt_hat[:,datapoint,0]
            EyEy = MC_pred[datapoint,0]**2
            Var[datapoint]= float(1/best_tau  + Eyy - EyEy)

        # Model Metrics
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5
        test_ll = sum(ll.squeeze())

        # Confidence Intervals
        on_target = 0
        ranges= []
        for mean,var,y in zip(MC_pred.squeeze(),Var,y_test):
            # Compute Interval
            conf_int= stats.norm.interval(1-alpha, loc=mean, scale= sqrt(var))
            # Update Metrics
            if (y>=conf_int[0] and y<=conf_int[1]): on_target+=1
            ranges.append( abs(conf_int[0]) + abs(conf_int[1]) )
        coverage = on_target/ len(ranges)
        avg_range = np.mean(ranges)

        # Done
        return(coverage, avg_range, test_ll)
