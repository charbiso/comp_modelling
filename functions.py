import numpy as np
def softmax_func(EVvec, indchoice):
    prob_opt = EVvec[indchoice] / np.sum(EVvec)
    return prob_opt

#softmax for use without index
def softmax(opt1, opt2):
    return np.exp(opt1) / (np.exp(opt1) + np.exp(opt2))

#softmax including temperature parameter
def softmaxw_temp(opt1, opt2, temp):
    if temp > 0:
        return np.exp(opt1 / temp) / (np.exp(opt1 / temp) + np.exp(opt2 / temp))

def compute_utility_risk_amb(alpha, beta, v, p, A):
    if A > 0 and p != 0.5:
        p = 0.5
        print("Warning: Non-zero ambiguity level specified. Risk level changed to 50%")
    if p > 1:
        return "Error: incorrect probability value"
    if A > 1:
        return "Error: incorrect uncertainty value"
    else:
        return (v ** alpha) * (p + ((beta * A) / 2))