# Numerical--> Categorical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import sklearn.cluster as sc
import functions
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric, NearestNeighbors


bioage_m = pd.read_csv("data/bioage_m_imputationR.csv")
bioage_f = pd.read_csv("data/bioage_f_R.csv")


# SBP

for i in bioage_m.index:
    if bioage_m.loc[i,"SBP"] < 120:
        bioage_m.loc[i,"SBP_1"] = "Negative"
        bioage_m.loc[i,"SBP_2"] = "Negative"
    elif 120 <= bioage_m.loc[i,"SBP"] and bioage_m.loc[i,"SBP"] < 140:
        bioage_m.loc[i,"SBP_1"] = "Positive"
        bioage_m.loc[i,"SBP_2"] = "Negative"
    elif 140 <= bioage_m.loc[i,"SBP"]:
        bioage_m.loc[i,"SBP_1"] = "Negative"
        bioage_m.loc[i,"SBP_2"] = "Positive"
    else:
        print("SBP_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"SBP"] < 120:
        bioage_f.loc[i,"SBP_1"] = "Negative"
        bioage_f.loc[i,"SBP_2"] = "Negative"
    elif 120 <= bioage_f.loc[i,"SBP"] and bioage_f.loc[i,"SBP"] < 140:
        bioage_f.loc[i,"SBP_1"] = "Positive"
        bioage_f.loc[i,"SBP_2"] = "Negative"
    elif 140 <= bioage_f.loc[i,"SBP"]:
        bioage_f.loc[i,"SBP_1"] = "Negative"
        bioage_f.loc[i,"SBP_2"] = "Positive"
    else:
        print("SBP_f", i)
        sys.exit()


print("SBP is FInished")

# DBP

for i in bioage_m.index:
    if bioage_m.loc[i,"DBP"] < 80:
        bioage_m.loc[i,"DBP_1"] = "Negative"
        bioage_m.loc[i,"DBP_2"] = "Negative"
    elif 80 <= bioage_m.loc[i,"DBP"] and bioage_m.loc[i,"DBP"] < 90:
        bioage_m.loc[i,"DBP_1"] = "Positive"
        bioage_m.loc[i,"DBP_2"] = "Negative"
    elif 90 <= bioage_m.loc[i,"DBP"]:
        bioage_m.loc[i,"DBP_1"] = "Negative"
        bioage_m.loc[i,"DBP_2"] = "Positive"
    else:
        print("DBP_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"DBP"] < 80:
        bioage_f.loc[i,"DBP_1"] = "Negative"
        bioage_f.loc[i,"DBP_2"] = "Negative"
    elif 80 <= bioage_f.loc[i,"DBP"] and bioage_f.loc[i,"DBP"] < 90:
        bioage_f.loc[i,"DBP_1"] = "Positive"
        bioage_f.loc[i,"DBP_2"] = "Negative"
    elif 90 <= bioage_f.loc[i,"DBP"]:
        bioage_f.loc[i,"DBP_1"] = "Negative"
        bioage_f.loc[i,"DBP_2"] = "Positive"
    else:
        print("DBP_f", i)
        sys.exit()


print("DBP is FInished")

# BFP

for i in bioage_m.index:
    if bioage_m.loc[i,"BFP"] < 20:
        bioage_m.loc[i,"BFP_1"] = "Negative"
        bioage_m.loc[i,"BFP_2"] = "Negative"
    elif 20 <= bioage_m.loc[i,"BFP"] and bioage_m.loc[i,"BFP"] < 25:
        bioage_m.loc[i,"BFP_1"] = "Positive"
        bioage_m.loc[i,"BFP_2"] = "Negative"
    elif 25 <= bioage_m.loc[i,"BFP"]:
        bioage_m.loc[i,"BFP_1"] = "Negative"
        bioage_m.loc[i,"BFP_2"] = "Positive"
    else:
        print("BFP_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"BFP"] < 30:
        bioage_f.loc[i,"BFP_1"] = "Negative"
        bioage_f.loc[i,"BFP_2"] = "Negative"
    elif 30 <= bioage_f.loc[i,"BFP"] and bioage_f.loc[i,"BFP"] < 35:
        bioage_f.loc[i,"BFP_1"] = "Positive"
        bioage_f.loc[i,"BFP_2"] = "Negative"
    elif 35 <= bioage_f.loc[i,"BFP"]:
        bioage_f.loc[i,"BFP_1"] = "Negative"
        bioage_f.loc[i,"BFP_2"] = "Positive"
    else:
        print("BFP_f", i)
        sys.exit()


print("BFP is FInished")

# BMI
for i in bioage_m.index:
    if 18.5 <= bioage_m.loc[i,"BMI"] and bioage_m.loc[i,"BMI"] < 25:
        bioage_m.loc[i,"BMI_1"] = "Negative"
        bioage_m.loc[i,"BMI_2"] = "Negative"
    elif bioage_m.loc[i,"BMI"] < 18.5:
        bioage_m.loc[i,"BMI_1"] = "Positive"
        bioage_m.loc[i,"BMI_2"] = "Negative"
    elif 25 <= bioage_m.loc[i,"BMI"] and bioage_m.loc[i,"BMI"] < 30:
        bioage_m.loc[i,"BMI_1"] = "Positive"
        bioage_m.loc[i,"BMI_2"] = "Negative"
    elif 30 <= bioage_m.loc[i,"BMI"]:
        bioage_m.loc[i,"BMI_1"] = "Negative"
        bioage_m.loc[i,"BMI_2"] = "Positive"
    else:
        print("BMI_m", i)
        sys.exit()

for i in bioage_f.index:
    if 18.5 <= bioage_f.loc[i,"BMI"] and bioage_f.loc[i,"BMI"] < 25:
        bioage_f.loc[i,"BMI_1"] = "Negative"
        bioage_f.loc[i,"BMI_2"] = "Negative"
    elif bioage_f.loc[i,"BMI"] < 18.5:
        bioage_f.loc[i,"BMI_1"] = "Positive"
        bioage_f.loc[i,"BMI_2"] = "Negative"
    elif 25 <= bioage_f.loc[i,"BMI"] and bioage_f.loc[i,"BMI"] < 30:
        bioage_f.loc[i,"BMI_1"] = "Positive"
        bioage_f.loc[i,"BMI_2"] = "Negative"
    elif 30 <= bioage_f.loc[i,"BMI"]:
        bioage_f.loc[i,"BMI_1"] = "Negative"
        bioage_f.loc[i,"BMI_2"] = "Positive"
    else:
        print("BMI_f", i)
        sys.exit()


print("BMI is FInished")


# FVCPP

for i in bioage_m.index:
    if bioage_m.loc[i,"FEV1_FVC"] >= 0.7:
        if 80 >= bioage_m.loc[i,"FVCPP"]:
            bioage_m.loc[i,"FVCPP_1"] = "Positive"
            bioage_m.loc[i,"FVCPP_2"] = "Negative"
        else: 
            bioage_m.loc[i,"FVCPP_1"] = "Negative"
            bioage_m.loc[i,"FVCPP_2"] = "Negative"
            
    if bioage_m.loc[i,"FEV1_FVC"] < 0.7:
        if 80 <= bioage_m.loc[i,"FVCPP"]:
            bioage_m.loc[i,"FVCPP_1"] = "Negative"
            bioage_m.loc[i,"FVCPP_2"] = "Positive"
        else: 
            bioage_m.loc[i,"FVCPP_1"] = "Positive"
            bioage_m.loc[i,"FVCPP_2"] = "Positive"


for i in bioage_f.index:
    if bioage_f.loc[i,"FEV1_FVC"] >= 0.7:
        if 80 >= bioage_f.loc[i,"FVCPP"]:
            bioage_f.loc[i,"FVCPP_1"] = "Positive"
            bioage_f.loc[i,"FVCPP_2"] = "Negative"
        else: 
            bioage_f.loc[i,"FVCPP_1"] = "Negative"
            bioage_f.loc[i,"FVCPP_2"] = "Negative"
            
    if bioage_f.loc[i,"FEV1_FVC"] < 0.7:
        if 80 <= bioage_f.loc[i,"FVCPP"]:
            bioage_f.loc[i,"FVCPP_1"] = "Negative"
            bioage_f.loc[i,"FVCPP_2"] = "Positive"
        else: 
            bioage_f.loc[i,"FVCPP_1"] = "Positive"
            bioage_f.loc[i,"FVCPP_2"] = "Positive"
    if bioage_f.loc[i,"FEV1_FVC"] == "NaN":
        print("FEV1_FVC", i)
        sys.exit()

print("FVCPP is FInished")

#FEV1PP


for i in bioage_m.index:
    if 40 <= bioage_m.loc[i,"FEV1PP"]:
        bioage_m.loc[i,"FEV1PP_1"] = "Negative"
        bioage_m.loc[i,"FEV1PP_2"] = "Negative"
        bioage_m.loc[i,"FEV1PP_3"] = "Negative"
    elif 30 <= bioage_m.loc[i,"FEV1PP"] and bioage_m.loc[i,"FEV1PP"] < 40:
        bioage_m.loc[i,"FEV1PP_1"] = "Positive"
        bioage_m.loc[i,"FEV1PP_2"] = "Negative"
        bioage_m.loc[i,"FEV1PP_3"] = "Negative"
    elif 25 <= bioage_m.loc[i,"FEV1PP"] and bioage_m.loc[i,"FEV1PP"] < 30:
        bioage_m.loc[i,"FEV1PP_1"] = "Negative"
        bioage_m.loc[i,"FEV1PP_2"] = "Positive"
        bioage_m.loc[i,"FEV1PP_3"] = "Negative"
    elif bioage_m.loc[i,"FEV1PP"] < 25:
        bioage_m.loc[i,"FEV1PP_1"] = "Negative"
        bioage_m.loc[i,"FEV1PP_2"] = "Negative"
        bioage_m.loc[i,"FEV1PP_3"] = "Positive"
    else:
        print("FEV1PP_f", i)
        sys.exit()

for i in bioage_f.index:
    if 40 <= bioage_f.loc[i,"FEV1PP"]:
        bioage_f.loc[i,"FEV1PP_1"] = "Negative"
        bioage_f.loc[i,"FEV1PP_2"] = "Negative"
        bioage_f.loc[i,"FEV1PP_3"] = "Negative"
    elif 30 <= bioage_f.loc[i,"FEV1PP"] and bioage_f.loc[i,"FEV1PP"] < 40:
        bioage_f.loc[i,"FEV1PP_1"] = "Positive"
        bioage_f.loc[i,"FEV1PP_2"] = "Negative"
        bioage_f.loc[i,"FEV1PP_3"] = "Negative"
    elif 25 <= bioage_f.loc[i,"FEV1PP"] and bioage_f.loc[i,"FEV1PP"] < 30:
        bioage_f.loc[i,"FEV1PP_1"] = "Negative"
        bioage_f.loc[i,"FEV1PP_2"] = "Positive"
        bioage_f.loc[i,"FEV1PP_3"] = "Negative"
    elif bioage_f.loc[i,"FEV1PP"] < 25:
        bioage_f.loc[i,"FEV1PP_1"] = "Negative"
        bioage_f.loc[i,"FEV1PP_2"] = "Negative"
        bioage_f.loc[i,"FEV1PP_3"] = "Positive"
    else:
        print("FEV1PP_f", i)
        sys.exit()

print("FEV1PP is FInished")

# AST

for i in bioage_m.index:
    if  bioage_m.loc[i,"AST"] < 40:
        bioage_m.loc[i,"AST_1"] = "Negative"
        bioage_m.loc[i,"AST_2"] = "Negative"
    elif 40 <= bioage_m.loc[i,"AST"] and bioage_m.loc[i,"AST"] < 50:
        bioage_m.loc[i,"AST_1"] = "Positive"
        bioage_m.loc[i,"AST_2"] = "Negative"
    elif 50 <= bioage_m.loc[i,"AST"]:
        bioage_m.loc[i,"AST_1"] = "Negative"
        bioage_m.loc[i,"AST_2"] = "Positive"
    else:
        print("AST_m", i)
        sys.exit()



for i in bioage_f.index:
    if bioage_f.loc[i,"AST"] < 40:
        bioage_f.loc[i,"AST_1"] = "Negative"
        bioage_f.loc[i,"AST_2"] = "Negative"
    elif 40 <= bioage_f.loc[i,"AST"] and bioage_f.loc[i,"AST"] < 50:
        bioage_f.loc[i,"AST_1"] = "Positive"
        bioage_f.loc[i,"AST_2"] = "Negative"
    elif 50 <= bioage_f.loc[i,"AST"]:
        bioage_f.loc[i,"AST_1"] = "Negative"
        bioage_f.loc[i,"AST_2"] = "Positive"
    else:
        print("AST_f", i)
        sys.exit()

print("AST is FInished")

# ALT


for i in bioage_m.index:
    if bioage_m.loc[i,"ALT"] < 35:
        bioage_m.loc[i,"ALT_1"] = "Negative"
        bioage_m.loc[i,"ALT_2"] = "Negative"
    elif 35 <= bioage_m.loc[i,"ALT"] and bioage_m.loc[i,"ALT"] < 45:
        bioage_m.loc[i,"ALT_1"] = "Positive"
        bioage_m.loc[i,"ALT_2"] = "Negative"
    elif 45 <= bioage_m.loc[i,"ALT"]:
        bioage_m.loc[i,"ALT_1"] = "Negative"
        bioage_m.loc[i,"ALT_2"] = "Positive"
    else:
        print("ALT_m", i)
        sys.exit()




for i in bioage_f.index:
    if bioage_f.loc[i,"ALT"] < 35:
        bioage_f.loc[i,"ALT_1"] = "Negative"
        bioage_f.loc[i,"ALT_2"] = "Negative"
    elif 35 <= bioage_f.loc[i,"ALT"] and bioage_f.loc[i,"ALT"] < 45:
        bioage_f.loc[i,"ALT_1"] = "Positive"
        bioage_f.loc[i,"ALT_2"] = "Negative"
    elif 45 <= bioage_f.loc[i,"ALT"]:
        bioage_f.loc[i,"ALT_1"] = "Negative"
        bioage_f.loc[i,"ALT_2"] = "Positive"
    else:
        print("ALT_f", i)
        sys.exit()


print("ALT is FInished")

# GGT

for i in bioage_m.index:
    if bioage_m.loc[i,"GGT"] < 64:
        bioage_m.loc[i,"GGT_1"] = "Negative"
        bioage_m.loc[i,"GGT_2"] = "Negative"
    elif 64 <= bioage_m.loc[i,"GGT"] and bioage_m.loc[i,"GGT"] < 78:
        bioage_m.loc[i,"GGT_1"] = "Positive"
        bioage_m.loc[i,"GGT_2"] = "Negative"
    elif 78 <= bioage_m.loc[i,"GGT"]:
        bioage_m.loc[i,"GGT_1"] = "Negative"
        bioage_m.loc[i,"GGT_2"] = "Positive"
    else:
        print("GGT_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"GGT"] < 36:
        bioage_f.loc[i,"GGT_1"] = "Negative"
        bioage_f.loc[i,"GGT_2"] = "Negative"
    elif 36 <= bioage_f.loc[i,"GGT"] and bioage_f.loc[i,"GGT"] < 46:
        bioage_f.loc[i,"GGT_1"] = "Positive"
        bioage_f.loc[i,"GGT_2"] = "Negative"
    elif 46 <= bioage_f.loc[i,"GGT"]:
        bioage_f.loc[i,"GGT_1"] = "Negative"
        bioage_f.loc[i,"GGT_2"] = "Positive"
    else:
        print("GGT_f", i)
        sys.exit()



print("GGT is FInished")

# CHOL
for i in bioage_m.index:
    if bioage_m.loc[i,"CHOL"] < 200:
        bioage_m.loc[i,"CHOL_1"] = "Negative"
        bioage_m.loc[i,"CHOL_2"] = "Negative"
    elif 200 <= bioage_m.loc[i,"CHOL"] and bioage_m.loc[i,"CHOL"] < 240:
        bioage_m.loc[i,"CHOL_1"] = "Positive"
        bioage_m.loc[i,"CHOL_2"] = "Negative"
    elif 240 <= bioage_m.loc[i,"CHOL"]:
        bioage_m.loc[i,"CHOL_1"] = "Negative"
        bioage_m.loc[i,"CHOL_2"] = "Positive"
    else:
        print("CHOL_m", i)
        sys.exit()



for i in bioage_f.index:
    if bioage_f.loc[i,"CHOL"] < 200:
        bioage_f.loc[i,"CHOL_1"] = "Negative"
        bioage_f.loc[i,"CHOL_2"] = "Negative"
    elif 200 <= bioage_f.loc[i,"CHOL"] and bioage_f.loc[i,"CHOL"] < 240:
        bioage_f.loc[i,"CHOL_1"] = "Positive"
        bioage_f.loc[i,"CHOL_2"] = "Negative"
    elif 240 <= bioage_f.loc[i,"CHOL"]:
        bioage_f.loc[i,"CHOL_1"] = "Negative"
        bioage_f.loc[i,"CHOL_2"] = "Positive"
    else:
        print("CHOL_f", i)
        sys.exit()



print("CHOL is FInished")

# HDL

for i in bioage_m.index:
    if 60 <= bioage_m.loc[i,"HDL"]:
        bioage_m.loc[i,"HDL_1"] = "Negative"
        bioage_m.loc[i,"HDL_2"] = "Negative"
    elif 40 <= bioage_m.loc[i,"HDL"] and bioage_m.loc[i,"HDL"] < 60:
        bioage_m.loc[i,"HDL_1"] = "Positive"
        bioage_m.loc[i,"HDL_2"] = "Negative"
    elif bioage_m.loc[i,"HDL"] < 40:
        bioage_m.loc[i,"HDL_1"] = "Negative"
        bioage_m.loc[i,"HDL_2"] = "Positive"
    else:
        print("HDL_m", i)
        sys.exit()



for i in bioage_f.index:
    if 60 <= bioage_f.loc[i,"HDL"]:
        bioage_f.loc[i,"HDL_1"] = "Negative"
        bioage_f.loc[i,"HDL_2"] = "Negative"
    elif 40 <= bioage_f.loc[i,"HDL"] and bioage_f.loc[i,"HDL"] < 60:
        bioage_f.loc[i,"HDL_1"] = "Positive"
        bioage_f.loc[i,"HDL_2"] = "Negative"
    elif bioage_f.loc[i,"HDL"] < 40:
        bioage_f.loc[i,"HDL_1"] = "Negative"
        bioage_f.loc[i,"HDL_2"] = "Positive"
    else:
        print("HDL_f", i)
        sys.exit()



print("HDL is FInished")

# TG

for i in bioage_m.index:
    if bioage_m.loc[i,"TG"] < 150:
        bioage_m.loc[i,"TG_1"] = "Negative"
        bioage_m.loc[i,"TG_2"] = "Negative"
    elif 150 <= bioage_m.loc[i,"TG"] and bioage_m.loc[i,"TG"] < 200:
        bioage_m.loc[i,"TG_1"] = "Positive"
        bioage_m.loc[i,"TG_2"] = "Negative"
    elif 200 <= bioage_m.loc[i,"TG"]:
        bioage_m.loc[i,"TG_1"] = "Negative"
        bioage_m.loc[i,"TG_2"] = "Positive"
    else:
        print("TG_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"TG"] < 150:
        bioage_f.loc[i,"TG_1"] = "Negative"
        bioage_f.loc[i,"TG_2"] = "Negative"
    elif 150 <= bioage_f.loc[i,"TG"] and bioage_f.loc[i,"TG"] < 200:
        bioage_f.loc[i,"TG_1"] = "Positive"
        bioage_f.loc[i,"TG_2"] = "Negative"
    elif 200 <= bioage_f.loc[i,"TG"]:
        bioage_f.loc[i,"TG_1"] = "Negative"
        bioage_f.loc[i,"TG_2"] = "Positive"
    else:
        print("TG_f", i)
        sys.exit()


print("TG is FInished")

# GLU
for i in bioage_m.index:
    if bioage_m.loc[i,"GLU"] < 100:
        bioage_m.loc[i,"GLU_1"] = "Negative"
        bioage_m.loc[i,"GLU_2"] = "Negative"
    elif 100 <= bioage_m.loc[i,"GLU"] and bioage_m.loc[i,"GLU"] < 126:
        bioage_m.loc[i,"GLU_1"] = "Positive"
        bioage_m.loc[i,"GLU_2"] = "Negative"
    elif 126 <= bioage_m.loc[i,"GLU"]:
        bioage_m.loc[i,"GLU_1"] = "Negative"
        bioage_m.loc[i,"GLU_2"] = "Positive"
    else:
        print("GLU_m", i)
        sys.exit()


for i in bioage_f.index:
    if bioage_f.loc[i,"GLU"] < 100:
        bioage_f.loc[i,"GLU_1"] = "Negative"
        bioage_f.loc[i,"GLU_2"] = "Negative"
    elif 100 <= bioage_f.loc[i,"GLU"] and bioage_f.loc[i,"GLU"] < 126:
        bioage_f.loc[i,"GLU_1"] = "Positive"
        bioage_f.loc[i,"GLU_2"] = "Negative"
    elif 126 <= bioage_f.loc[i,"GLU"]:
        bioage_f.loc[i,"GLU_1"] = "Negative"
        bioage_f.loc[i,"GLU_2"] = "Positive"
    else:
        print("GLU_f", i)
        sys.exit()



print("GLU is FInished")

# HB
for i in bioage_m.index:
    if 13 <= bioage_m.loc[i,"HB"] and bioage_m.loc[i,"HB"] < 16.5:
        bioage_m.loc[i,"HB_1"] = "Negative"
        bioage_m.loc[i,"HB_2"] = "Negative"
    elif 12 <= bioage_m.loc[i,"HB"] and bioage_m.loc[i,"HB"] < 13:
        bioage_m.loc[i,"HB_1"] = "Positive"
        bioage_m.loc[i,"HB_2"] = "Negative"
    elif 16.5 <= bioage_m.loc[i,"HB"] and bioage_m.loc[i,"HB"] < 17.5:
        bioage_m.loc[i,"HB_1"] = "Positive"
        bioage_m.loc[i,"HB_2"] = "Negative"
    elif bioage_m.loc[i,"HB"] < 12:
        bioage_m.loc[i,"HB_1"] = "Negative"
        bioage_m.loc[i,"HB_2"] = "Positive"
    elif bioage_m.loc[i,"HB"] >= 17.5:
        bioage_m.loc[i,"HB_1"] = "Negative"
        bioage_m.loc[i,"HB_2"] = "Positive"
    else:
        print("HB_m", i)
        sys.exit()


for i in bioage_f.index:
    if 12 <= bioage_f.loc[i,"HB"] and bioage_f.loc[i,"HB"] < 15.5:
        bioage_f.loc[i,"HB_1"] = "Negative"
        bioage_f.loc[i,"HB_2"] = "Negative"
    elif 10 <= bioage_f.loc[i,"HB"] and bioage_f.loc[i,"HB"] < 12:
        bioage_f.loc[i,"HB_1"] = "Positive"
        bioage_f.loc[i,"HB_2"] = "Negative"
    elif 15.5 <= bioage_f.loc[i,"HB"] and bioage_f.loc[i,"HB"] < 16.5:
        bioage_f.loc[i,"HB_1"] = "Positive"
        bioage_f.loc[i,"HB_2"] = "Negative"
    elif bioage_f.loc[i,"HB"] >= 16.5:
        bioage_f.loc[i,"HB_1"] = "Negative"
        bioage_f.loc[i,"HB_2"] = "Positive"
    elif bioage_f.loc[i,"HB"] < 10:
        bioage_f.loc[i,"HB_1"] = "Negative"
        bioage_f.loc[i,"HB_2"] = "Positive"
    else:
        print("HB_f", i)
        sys.exit()


print("HB is FInished")

# Drop Vars
bioage_m = bioage_m.drop(columns=['SBP', 'DBP', 'BFP', 'BMI', 'FVCPP', 'FEV1PP','FEV1_FVC',
                       'AST', 'ALT', 
                       'GGT', 'CHOL', 'HDL', 'TG', 'GLU', 'HB'])
bioage_f = bioage_f.drop(columns=['SBP', 'DBP', 'BFP', 'BMI', 'FVCPP', 'FEV1PP','FEV1_FVC',
                       'AST', 'ALT',
                       'GGT', 'CHOL', 'HDL', 'TG', 'GLU', 'HB'])




outfile = open("data/bioage_m_change",'wb')
pickle.dump(bioage_m,outfile)
outfile.close()
outfile2 = open("data/bioage_f_change",'wb')
pickle.dump(bioage_f,outfile2)
outfile2.close()
