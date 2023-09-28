'''Tutorial adapted from https://github.com/psychNerdJae/cog-comp-modeling
for python (original tutorial written for R'''
from functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

'''Tutorial part 1: Expected values and softmax functions'''

# Expected value = value v x probability p
# Choice rules
## Hardmax function - choose which option has higher EV
## Softmax function - choose in proportion to EV
## Where p E [0,1], p(choose X) = EV(X)/EV(X) + EV(Y)

# Exercise 1: Calculate e^x, where x=[-5,5] and plot.

range1 = range(-5, 6, 1)
results = [np.exp(x) for x in range1]
plt.plot(results)
plt.show()

# Exercise 2: Softmax function.
''' a) Confirm the probability of choosing opt1 over opt2 is .73, 
given that EVopt1 = 4 and EVopt2 = 3'''

EVopt1 = 4
EVopt2 = 3
prob_opt1 = np.exp(EVopt1) / (np.exp(EVopt1) + np.exp(EVopt2))

''' b) Write a custom function implementing the softmax function. It should output
a single probability of choosing the given option.  As input, you should specify a 
vector of values (e.g. EVs), and also the index of the option being chosen. '''

EVopt1 = 4
EVopt2 = 3
EVopt3 = -2
EVopt4 = 1
EVvec = np.exp(np.array([EVopt1, EVopt2, EVopt3, EVopt4]))
indchoice = 0
result = softmax_func(EVvec, indchoice)

# result = .41 when EVopt = 4

'''c) Assume that EVopt3 = -2, how does this change the probability of choosing opt3?'''

# result = .002 when EVopt = -2

'''d) Imagine that we scale all of our EVs by a factor
of 10, such that EVopt1 = 40 and EVopt2 = 30. What is 
your expectation of how this will affect the probability
of choosing opt1? Use the function to compute
the probability'''

# When opt1 = 4, prob = .70
EVopt1 = 40
EVopt2 = 30
EVopt3 = -20
EVopt4 = 10
EVvec = np.exp(np.array([EVopt1, EVopt2, EVopt3, EVopt4]))
indchoice = 0
result2 = softmax_func(EVvec, indchoice)
# When opt1 = 40, prob = .999

'''Try plotting the probability of choosing opt1 over opt2, as the EVs of opt1 and opt2 change
as a heatmap, where the range of EVs for each option ranges from x E [-2,2]'''

opt1 = range(-2, 3)
opt2 = range(-2, 3)

prob_results = []
for x in opt1:
    for y in opt2:
        probOpt1 = softmax(x, y)
        prob_results.append({'opt1': x, 'opt2': y, 'probOpt1': probOpt1})

prob_df = pd.DataFrame(prob_results)

# heatmap_data = prob_df.pivot_table(index='opt1', columns='opt2', values='probOpt1', aggfunc='mean')
#
# sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", cbar=True)
# plt.xlabel('opt2')
# plt.ylabel('opt1')
# plt.title('Heatmap of ProbOpt1')
# plt.show()

'''e) If the values being passed into the softmax are quite large (or small), you may want to consider re-scaling them
into a range where the softmax is better behaved.
One common method is using a "temperature" parameter. For t e (0, inf], 
p(choose x1) = e^(x1/t) / sum(k to j=1)(e^x2/t)'''

opt1 = range(-2, 3)
opt2 = range(-2, 3)
temp = 2

prob_results = []
for x in opt1:
    for y in opt2:
        probOpt1 = softmaxw_temp(x, y, temp)
        prob_results.append({'opt1': x, 'opt2': y, 'probOpt1': probOpt1})

prob_df = pd.DataFrame(prob_results)

# heatmap_data = prob_df.pivot_table(index='opt1', columns='opt2', values='probOpt1', aggfunc='mean')
#
# sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", cbar=True)
# plt.xlabel('opt2')
# plt.ylabel('opt1')
# plt.title('Heatmap of ProbOpt1 with temperature')
# plt.show()

'''Tutorial part 2: model simulation'''
'''UTILITY: The EV equation can be modified to account for risk
This is called the utility: U = v^alpha x p, where alpha is a risk aversion parameter 
Then we can incorporate uncertainty U(alpha,beta;v,p,A) = v^alpha x (p + betaA/2) where uncertainty is A and beta is
a parameter than captures aversion to uncertainty. Dividing by 2 makes uncertainty symmetrical'''

# Exercise 2: Utility functions

'''a) Try implementing a custom utility function, that checks that the input values make sense'''

utility = compute_utility_risk_amb(alpha = 1, beta = -1, v = 25, p = .5, A = 0)





