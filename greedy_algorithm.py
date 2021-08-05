#!/usr/bin/env python
# coding: utf-8

# In[28]:


#minimization function
import random
def test_function(list3):
    i = 8
    for index in list3:
        i = i+2
        
    return (i * random.random())


# In[29]:


list3 = [2, 3, 4, 5, 6]
test_function(list3)


# In[30]:


#.....Coder : Anusuya Ghosh.....
#Date....4th August 2021......
#Time....9PM...IST...
#Implementation of Greedy Heuristics Search.............
#...For Level 2 : Micro Optimization...
#...updated with function at 5:40PM..............


import pandas as pd
import numpy as np
import array
import random
from pathlib import Path
import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter
from functools import reduce
import random
import itertools
from sklearn.base import BaseEstimator, TransformerMixin
import operator
import re
import xlrd
import pickle
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import warnings
import sklearn

sklearn.__version__
warnings.filterwarnings('ignore')
sns.set()
plt.rcParams["axes.grid"] = False
random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')

#....Defining Greedy Algorithm as a function..........

def greedy_algorithm(func, n: int, k1: int, *arguments_b): #input : objective function,
#number of technicians, technician's route, iterations, time windows for each customer of the route

    lb = [None]*n
    ub = [None]*n
    initial_point = [None]*n
    t = [None]*n
    bestpoints = [None]*n
    initial_point_v =[ [ None for i in range(k1-1) ] for j in range(n) ]
    #print(initial_point_v)

    h=0
    for arg in arguments_b:
        h = h+1
    #print("No of arguments:", h)


    
    i = 0
    for variable in range(0, n):
        if lb[variable] == None:
            lb[variable] = arguments_b[i]
            i = i+1
        #print("Lower Bounds:", lb[variable])

    m = h//2 
    for variable in range(0, n):
        if ub[variable] == None:
            ub[variable] = arguments_b[m]
            m = m+1
        #print("Upper Bounds:", ub[variable])

    for variable in range(0, n):
        initial_point[variable] = (ub[variable] + lb[variable])//2
        #print("Initial Points:", initial_point[variable])

    
    for inc in range(0, n):
        bestpoints[inc] = initial_point[inc]
        #print("Best Points:", bestpoints[inc])

    for i in range(0,n):
        for j in range(0,(k1-1)):
            initial_point_v[i][j] = initial_point[i]
            #print("Initial Points v:", i, j, initial_point_v[i][j])
            
    all_t = []
    for index in range(0, n):
        t[index] = bestpoints[index]
        all_t = [t[index] for index in range(0, n)]
        #print("Print all t:", all_t)

#-------loop starts here---------------------------------------------------------------------                
    i = 0
    while(i < n):
        k = 1
        update = 0
        print("Print all_t:", all_t)
        f_i_update = func( all_t)
        #print("Print all t after", all_t)
        #print("Print the function value f_i_update:", f_i_update)
        f_i_k = f_i_update                       

        while(k < k1):
            print("The iteration:", k, "The variable:", i)
            initial_point_v[i][k-1]= initial_point_v[i][update] + ((ub[i] - lb[i])*(random.uniform(0, 1) - 0.5))
            print("Print the initial_point_v:", initial_point_v[i][k-1], i, k-1)

            if((initial_point_v[i][k-1] >= lb[i]) and (initial_point_v[i][k-1] <= ub[i])):
                t[i] = initial_point_v[i][k-1]
                updated_all_t = [t[i] for i in range(0, n)]
                print("All the updated t values:", updated_all_t)
                f_i_k = func(updated_all_t)
                print("The function value f_i_k:", f_i_k, i, k)
                print("The function value f_i-update:", f_i_update, i, update)
            if(f_i_k < f_i_update):
                update = k
                print("The update is:", update)
                f_i_update = f_i_k
                bestpoints[i] = initial_point_v[i][k-1]
                print("The bestpoint in this case:", bestpoints[i], i)
            k = k + 1
            t[i] = bestpoints[i]
            print("The new updated t based on bestpoints:", t[i], i)
        i = i + 1
    return bestpoints 
        
greedy_algorithm(test_function, 4, 5, 23, 23, 65, 65, 29, 29, 75, 75) 


# In[ ]:




