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

'''Simulate how alpha affects utility during risky gambles. Use the following parameter space:
- Gamble = £100
- Risk levels .25, .5, and .75
- Ambiguity levels = 0
- alpha E [-.5, 1.2] in steps of .01
- beta can be fixed at arbitrary value'''

new_alpha = np.arange(-.5,1.21, .01)
win_probability = [.25, .5, .75]


risky_gamble_utility = []
for x in new_alpha:
    for y in win_probability:
        gamble_utility = compute_utility_risk_amb(alpha=x, beta=.5, v = 100, p=y, A = 0)
        risky_gamble_utility.append({'new_alpha': x, 'win_probability': y, 'gamble_utility': gamble_utility})

utility_df = pd.DataFrame(risky_gamble_utility)

utility_df['new_alpha'] = utility_df['new_alpha'].round(2)

heatmap_data = utility_df.pivot_table(index='win_probability', columns='new_alpha', values='gamble_utility', aggfunc='mean')
# fix axes
x_ticks = np.arange(0, len(heatmap_data.columns), 10)
x_labels = heatmap_data.columns[::10]
y_ticks = np.arange(0, len(heatmap_data.index), 1)
y_labels = heatmap_data.index

# make heatmap in matplotlib
# (Seaborn was unable to make this heatmap)
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', extent=[0, len(new_alpha), 0, len(win_probability)])
plt.colorbar(label='Gamble Utility')

plt.xticks(x_ticks, x_labels)
plt.yticks(y_ticks, y_labels)

plt.xlabel('Alpha')
plt.ylabel('Gamble win probability')
plt.title('Heatmap of Utility under Risk')
plt.show()


'''Now, simulate how beta affects utility during ambiguous gambles using the following parameter space:
- Gamble = £100
- Risk = .5
- Ambiguity = .25, .5, .75
- alpha E [.5, 1] in steps of .01
- beta E [-2, 2] in steps of .01'''

ambiguity = [.25,.5,.75]
alpha2 = np.arange(.5, 1.01, .01)
beta2 = np.arange(-2, 2.01, .01)




fig, axes = plt.subplots(nrows = 1, ncols = len(ambiguity), figsize=(15,5))

for i, A_value in enumerate(ambiguity):
    plot1 = []
    for x in alpha2:
        for y in beta2:
            gamble_utility = compute_utility_risk_amb(alpha=x, beta=y, v=100, p=.5, A=A_value)
            plot1.append({'alpha2': x, 'beta2': y, 'gamble_utility': gamble_utility, 'ambiguity': A_value})


    Amb_df = pd.DataFrame(plot1)
    Amb_df['alpha2'] = Amb_df['alpha2'].round(2)
    Amb_df['beta2'] = Amb_df['beta2'].round(2)
    heatmap_data = Amb_df.pivot_table(index='alpha2', columns='beta2', values='gamble_utility', aggfunc='mean')

    im = axes[i].imshow(heatmap_data, cmap='viridis', aspect='auto', extent=[0, len(alpha2), 0, len(beta2)])
    axes[i].set_title(f'A = {A_value}')
    # axes[i].set_xticks(np.arange(0, len(heatmap_data.columns),10))
    # axes[i].set_xticklabels(heatmap_data.columns[::10])
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Beta')
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=.02, pad=.1, label = 'Utility')
plt.suptitle('Heatmap of Utility under Risk for Different Ambiguity Levels')
plt.show()






