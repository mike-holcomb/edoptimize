# Estimate of Distribution Optimizer
**Michael Holcomb**

An example of an training algorithm for neural network algorithms using 
an estimation of distribution non-convex optimization algorithm.

If we assume that an optimal set of network parameters belongs to a multivariate 
normal distribution, we can iteratively refine our estimate of that distribution
 using a genetic algorithm heuristic.

In pseudocode:
```
X    = training inputs
y    = training outputs
F(x) = network to optimize
w_i  = vector of parameters of network i
n    = population size
m    = iterations
f    = fraction of population to keep
 
edoptimize(X, y, F(x), n, m, )
   W <- instantiate n copies of the network F(x) with uniformly distributed weights
   for m times:
       O <- forward_pass(W, X)       # Estimate y using all n networks
       E <- calculate_error(O, y)    # Calculate error of all n predictions
       B <- select n*f best networks
       mu, corr <- Calculate the average and correlation of w_i for all i in B
       W <- instantiate (1-f)*n copies of the network by sampling weights from 
            the multivariate distribion N(mu, corr)
       W <- W union B
   end for
   
   j <- index of W corresponding to network with smallest training error
   
   return w_i
end edooptimize
```

As with most heuristics, there are a number of hyperparameters to select.  Also a 
potentially better end condition would relate to the algorithms convergence (i.e.,
 checking a threshold with respect to the parameter variance, change in average 
 error or change in minimum error)
