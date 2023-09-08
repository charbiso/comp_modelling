'''Tutorial adapted from https://github.com/psychNerdJae/cog-comp-modeling
for python (original tutorial written for R'''

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Expected value = value v x probability p
# Choice rules
## Hardmax function - choose which option has higher EV
## Softmax function - choose in proportion to EV
## Where p E [0,1], p(choose X) = EV(X)/EV(X) + EV(Y)

# Exercise 1: Calculate e^x, where x=[-5,5] and plot.

range1 = [*range(-5, 6, 1)]
results = [math.exp(x) for x in range1]
plt.plot(results)
plt.show()

# Exercise 2: Softmax function.
''' a) Confirm the probability of choosing opt1 over opt2 is .73, 
given that EVopt1 = 4 and EVopt2 = 3'''


EVopt1 = 4
EVopt2 = 3
prob_opt1 = math.exp(EVopt1) / (math.exp(EVopt1) + math.exp(EVopt2))

''' b) Write a custom function implementing the softmax function. It should output
a single probability of choosing the given option.  As input, you should specify a 
vector of values (e.g. EVs), and also the index of the option being chosen. '''

def softmax_func(EVvec, indchoice):
    prob_opt = EVvec[indchoice]/np.sum(EVvec)
    return prob_opt

EVopt1 = 4
EVopt2 = 3
EVopt3 = -2
EVopt4 = 1
EVvec = np.exp(np.array([EVopt1, EVopt2, EVopt3, EVopt4]))
indchoice = 0
result = softmax_func(EVvec, indchoice)


#result = .41 when EVopt = 4

'''c) Assume that EVopt3 = -2, how does this change the probability of choosing opt3?'''

#result = .002 when EVopt = -2

'''d) Imagine that we scale all of our EVs by a factor
of 10, such that EVopt1 = 40 and EVopt2 = 30. What is 
your expectation of how this will affect the probability
of choosing opt1? Use the function to compute
the probability'''

#When opt1 = 4, prob = .70
EVopt1 = 40
EVopt2 = 30
EVopt3 = -20
EVopt4 = 10
EVvec = np.exp(np.array([EVopt1, EVopt2, EVopt3, EVopt4]))
indchoice = 0
result2 = softmax_func(EVvec, indchoice)
print(result2)
#When opt1 = 40, prob = .999

'''Try plotting the probability of choosing opt1 over opt2, as the EVs of opt1 and opt2 change
as a heatmap, where the range of EVs for each option ranges from x E [-2,2]'''

opt1 = [*range(-2,3,1)]
opt2 = [*range(-2,3,1)]
EVsopts = np.exp(np.array([opt1,opt2]))
result_soft = softmax_func(EVsopts,0)

plt.imshow(result_soft, cmap='hot', interpolation = 'nearest')
plt.show()















