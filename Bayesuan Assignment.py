# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 00:26:21 2024

@author: Diego Emmanuel C Cordero
"""

import matplotlib.pyplot as plt
import numpy as np

true_slope = 5
true_intercept = 10
true_sigma = 1 

num_points = 100

x_vals = np.linspace(0, 1, num_points)
true_y_vals = true_slope * x_vals + true_intercept
y_vals = true_y_vals + np.random.normal(scale= true_sigma, size= num_points)

true_params = {'slope': true_slope,'intercept': true_intercept, 'sigma':true_sigma}

plt.figure(figsize=(7,7))
p1 = plt.scatter(x_vals, y_vals)
p2 = plt.plot(x_vals, true_y_vals, color="r")
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

linearRegression = ()
clf = linearRegression 
clf.fit(x_vals.reshape(-1,1), y_vals)
preds = clf.predict(x_vals.reshape(-1, 1))
resids = preds - y_vals

print('True Model')
print('y_true =%s*x + %s' %(true_slope, true_intercept))
print('true sigma: %s/n' %(true_params['sigma']))

print('Estimated Model')
print('y_hat = %s*x + %s' %(clf.coef[0], clf.intercept_))
print('Sd Residuals: %s'%(resids.std()))

mle_estimates ={'slope': clf.coef_[0], 'intercept': clf.intercept_, 'sigma': resids.std()}

import pymc3 as pm

with pm.model() as model:
    #prior
    Sigma = pm.Exponential('sigma', lam=1.0)
    intercept = pm.normal('intercept', mu=0, sigma=20)
    slope = pm.normal('slope',  mu=0, sigma=20)
    
    #likelihood 
    likelihood = pm.normal('y', mu=slope*x_vals + intercept, sigma=Sigma, observered=y_vals)
    
    #posterior
    trace = pm.sample(1000, core=4)
    
