#!/usr/bin/env python
# coding: utf-8
 Let take the example from th cost function eqution for the linear model where as Q0=0
    
    i.e: j(Q1)=1/2m(∑i=1,m(hθ(X(i))−Y(i))2
# In[2]:


##Import some of the important libraries


# In[34]:


import numpy as np
import matplotlib.pyplot as plt


# In[43]:


# X = np.array([[1], [2], [3]])
# y = np.array([[1], [2.5], [3.5]])

# get_theta = lambda theta: np.array([[0, theta]])

# thetas = list(map(get_theta, [0.5, 1.0, 1.5]))

# X = np.hstack([np.ones([3, 1]), X])

# def cost(X, y, theta):
#     inner = np.power(((X @ theta.T) - y), 2)
#     return np.sum(inner) / (2 * len(X))

# for i in range(len(thetas)):
#     print(cost(X, y, thetas[i]))


# Calculating the cost function using Python

# Remember  we are finding the difference between estimated values (Y), or the difference between the hypothesis and the real values — the actual data we are trying to fit a line to.

# ╔═══════╦═══════╦═════════════╗
# ║   X   ║ y     ║  best_fit_1 ║
# ╠═══════╬═══════╬═════════════╣
# ║ 1.00  ║ 1.00  ║    0.50     ║
# ║ 2.00  ║ 2.50  ║    1.00     ║
# ║ 3.00  ║ 3.50  ║    1.50     ║
# ╚═══════╩═══════╩═════════════╝

# In[103]:


# # original data set
X = [1, 2, 3]
y = [1,2.5,3.5]

# # Another data set
# X = [1, 2, 3]
# y = [1.7,3.5,5]


# In[104]:


gr=plt.scatter(X,y)
gr=plt.plot(X,y)


# In[107]:



# slope of best_fit_1 is 1.083
# slope of best_fit_2 is 0.083
# slope of best_fit_3 is 0.25

# hyps =[1.083]
hyps =[1.083, 0.083, 0.25]


# mutiply the original X values by the theta 
# to produce hypothesis values for each X
def multiply_matrix(mat, theta):
    mutated = []
    for i in range(len(mat)):
        mutated.append(mat[i] * theta)

    return mutated

# calculate cost by looping each sample
# subtract hyp(x) from y
# square the result
# sum them all together
def calc_cost(m, X, y):
    total = 0
    for i in range(m):
        squared_error = (y[i] - X[i]) ** 2
        total += squared_error
    
    return total * (1 / (2*m))

# calculate cost for each hypothesis
for i in range(len(hyps)):
    hyp_values = multiply_matrix(X, hyps[i])

    print("Cost for ", hyps[i], " is ", calc_cost(len(X), y, hyp_values))


# In[ ]:




