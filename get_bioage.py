import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import sklearn.cluster as sc
import time
import functions
from tabulate import tabulate
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


infile = open("data/bioage_change_divided_group1_compare1_group2_compare2",'rb')
clusters = pickle.load(infile)
infile2 = open("data/get_index",'rb')
sets = pickle.load(infile2)


m4 = sets[0]; m5 = sets[1]; f4 = sets[2]; f5 = sets[3]
male_index = list(m4.index) + list(m5.index)
female_index = list(f4.index) + list(f5.index)


m_4 = clusters[0]; m_5 = clusters[1]; f_4 = clusters[2]; f_5 = clusters[3]
m_4_body = m_4[0:4]; m_4_lung = m_4[4:8]; m_4_blood = m_4[8:12]; m_4_res = m_4[12:16]; m_4_urine = m_4[16:20]
m_5_body = m_5[0:4]; m_5_lung = m_5[4:8]; m_5_blood = m_5[8:12]; m_5_res = m_5[12:16]; m_5_urine = m_5[16:20]
f_4_body = f_4[0:4]; f_4_lung = f_4[4:8]; f_4_blood = f_4[8:12]; f_4_res = f_4[12:16]; f_4_urine = f_4[16:20]
f_5_body = f_5[0:4]; f_5_lung = f_5[4:8]; f_5_blood = f_5[8:12]; f_5_res = f_5[12:16]; f_5_urine = f_5[16:20]


# Male
def male_bioage(group_4, group_5):
    bioage = {}
    for idx in male_index:
        if idx in group_4[0].index:
            bioage[idx] = 62
        elif idx in group_4[1].index:
            bioage[idx] = 64.5
        elif idx in group_4[2].index:
            bioage[idx] = 67
        elif idx in group_4[3].index:
            bioage[idx] = 69
        elif idx in group_5[0].index:
            bioage[idx] = 72
        elif idx in group_5[1].index:
            bioage[idx] = 74.5
        elif idx in group_5[2].index:
            bioage[idx] = 77
        elif idx in group_5[3].index:
            bioage[idx] = 79
        else:
            print("Error")
            sys.exit()
    return bioage

# Female
def female_bioage(group_4, group_5):
    bioage = {}
    for idx in female_index:
        if idx in group_4[0].index:
            bioage[idx] = 62
        elif idx in group_4[1].index:
            bioage[idx] = 64.5
        elif idx in group_4[2].index:
            bioage[idx] = 67
        elif idx in group_4[3].index:
            bioage[idx] = 69
        elif idx in group_5[0].index:
            bioage[idx] = 72
        elif idx in group_5[1].index:
            bioage[idx] = 74.5
        elif idx in group_5[2].index:
            bioage[idx] = 77
        elif idx in group_5[3].index:
            bioage[idx] = 79
        else:
            print("Error")
            sys.exit()
    return bioage

# Get dictionary of bioage
male_bioage_body = male_bioage(m_4_body, m_5_body)
male_bioage_lung = male_bioage(m_4_lung, m_5_lung)
male_bioage_blood = male_bioage(m_4_blood, m_5_blood)
male_bioage_res = male_bioage(m_4_res, m_5_res)
male_bioage_urine = male_bioage(m_4_urine, m_5_urine)
male_bioage_ls = [male_bioage_body, male_bioage_blood, male_bioage_res, male_bioage_urine]
#male_bioage_ls = [male_bioage_body, male_bioage_lung, male_bioage_blood, male_bioage_res, male_bioage_urine]

female_bioage_body = female_bioage(f_4_body, f_5_body)
female_bioage_lung = female_bioage(f_4_lung, f_5_lung)
female_bioage_blood = female_bioage(f_4_blood, f_5_blood)
female_bioage_res = female_bioage(f_4_res, f_5_res)
female_bioage_urine = female_bioage(f_4_urine, f_5_urine)
female_bioage_ls = [female_bioage_body, female_bioage_blood, female_bioage_res, female_bioage_urine]
#female_bioage_ls = [female_bioage_body, female_bioage_lung, female_bioage_blood, female_bioage_res, female_bioage_urine]


# Get bioage dictionary
bioage_male = {}
for idx in male_index:
    bioage_tmp = 0
    for dic_var in male_bioage_ls:
        bioage_tmp = bioage_tmp + dic_var[idx]
    bioage_male[idx] = bioage_tmp/4

bioage_female = {}
for idx in female_index:
    bioage_tmp = 0
    for dic_var in female_bioage_ls:
        bioage_tmp = bioage_tmp + dic_var[idx]
    bioage_female[idx] = bioage_tmp/4
    
print("Making bioage")
bioage = [bioage_male, bioage_female]
outfile = open("data/estimated_bioage",'wb')
pickle.dump(bioage,outfile)
outfile.close()

bioage_vars = [male_bioage_ls, female_bioage_ls]
outfile2 = open("data/estimated_variable_bioage",'wb')
pickle.dump(bioage_vars,outfile2)
outfile.close()









