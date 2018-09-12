import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import statsmodels
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

bioage = pd.read_csv("data/bioage.csv")

# <1> NEUTROPHIL, LYMPHOCYT, MONOCYTE, EOSINOPHIL, BASOPHIL = 100
dic = {}
ls = []
for r in range(bioage.shape[0]):
    k = bioage.iloc[r,51:56].isnull()
    sum = 0
    for j in k:
        if j:
            sum = sum + 1
    dic[r] = sum
    if sum == 1:
        ls.append(r)
        
# BASOPHIL = 0 --> NaN 이므로 NaN을 0으로 바꿔줌
ls2 = []
for i in ls:
    if bioage.iloc[i,51:55].sum() > 99.9:
        bioage.set_value(i,bioage.columns[55],0) == 0
        ls2.append(i)
        
for i in ls2:
    if i in ls:
        ls.remove(i)
        
        
# MONOCYTE = 0 --> NaN 이므로 NaN을 0으로 바꿔줌
for i in ls:
    bioage.set_value(i,bioage.columns[53],0) == 0
    
bioage.to_csv("bioage_V1.csv")
outfile = open("data/bioage_V1",'wb')
pickle.dump(bioage,outfile)
outfile.close()



