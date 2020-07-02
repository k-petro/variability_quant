# Variability Quantification in Machine Learning
Research Project on variability quantification in Machine Learning and Deep Learning.

The repository contains the code for the experimental procedures of the homonymous paper. 
The performance and the resulting prediction intervals are examined for the following methods:
1. __NGBoost__: Probabilistic alternative to _Gradient Boosting_. [Duan et. al](https://stanfordmlgroup.github.io/projects/ngboost/)
2. __subsampled Random Forest__: Random Forest incorporating a U-Statistic structure. [Mentch & Hooker](https://arxiv.org/pdf/1404.6473.pdf)
3. __MC-Dropout__: Randomness induced in Neural Network prediction as a Bernoulli behind each weight. [Gal & Ghahramani](http://proceedings.mlr.press/v48/gal16.pdf)

5 standard regression datasets from the UCI database are used to compare the performance of the models above. 
To test, a k-fold split is implemented and the test results are averaged across folds, for each dataset.
The performance comparison is evaluated on the test log-likelihood under each model. 
Furthermore, prediction intervals are constructed from each model and evaluated on 2 metrics. 

The datasets options are: `concrete, energy, housing, wine, yacht`.
The model options are: `ngboost, subRF, mc`.
Alpha is the parameter used to define confidence level := 1-α
The experiment is done by executing in shell the following command:
`./test_model -data <dataset_option> -nf <number_of_testing_folds> -model <model_option> -a <alpha>`


