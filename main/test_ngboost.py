import ngboost
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
import math
import numpy as np
from scipy import stats

# Define Gaussian Log-Likelihood Function
point_ll= lambda data, loc, scale: -.5*math.log(2*math.pi*(scale**2))/2 - ((data-loc)**2)/(2 * (scale**2))
ll = np.vectorize(point_ll)

def test_ngboost(X_train, y_train, X_val, y_val, X_test, y_test, alpha):
    # Parameter Grid Search
    b5 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
    b10 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=10)
    b15 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=15)
    b20 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=20)
    b25 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=25)
    b30 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=30)
    b35 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=35)

    param_grid = {
        'minibatch_frac': [1.0, 0.75, 0.5],
        'Base': [b5, b10, b15, b20, b25, b30, b35],
        'learning_rate': [1e-3,5e-3,1e-2,5e-2,.1]
    }


    model = ngboost.NGBRegressor(Dist=Normal, Score= LogScore, verbose=False)
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Obtain optimum parameters from grid search and train model with validaton score
    best_model = ngboost.NGBRegressor(Dist=Normal, verbose=False, 
                                      Base= grid_search.best_params_['Base'],
                                      minibatch_frac= grid_search.best_params_['minibatch_frac'],
                                      learning_rate= grid_search.best_params_['learning_rate'],
                                     ).fit(X_train, y_train, X_val, y_val, early_stopping_rounds=40)

    # Test
    y_pred = best_model.pred_dist(X_test,max_iter=best_model.best_val_loss_itr)

    # Model Metrics
    rmse = np.mean((y_test - y_pred.params['loc'])**2.)**0.5
    test_ll= sum(ll(y_test, y_pred.params['loc'], y_pred.params['scale'] ))

    # Confidence Intervals
    on_target = 0
    ranges= []
    for loc,scale,y in zip(y_pred.params['loc'], y_pred.params['scale'], y_test):
        # Compute Interval
        conf_int= stats.norm.interval(1-alpha, loc=loc, scale= scale)
        # Update Metrics
        if (y>=conf_int[0] and y<=conf_int[1]): on_target+=1
        ranges.append( abs(conf_int[0]) + abs(conf_int[1]) )
    coverage = on_target/ len(ranges)
    avg_range = np.mean(ranges)

    # Done.
    return( coverage, avg_range, test_ll )