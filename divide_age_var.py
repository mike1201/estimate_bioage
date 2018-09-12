import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import sklearn.cluster as sc
import time
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric, NearestNeighbors

infile = open("data/bioage_m_change",'rb')
bioage_m = pickle.load(infile)
bioage_m.index = bioage_m.iloc[:,0]
bioage_m = bioage_m.iloc[:,3:] # Get row name and erase unnamed column

infile2 = open("data/bioage_f_change",'rb')
bioage_f = pickle.load(infile2)
bioage_f.index = bioage_f.iloc[:,0]
bioage_f = bioage_f.iloc[:,3:]

print("--------------------------------------")
print("Import bioage : numerical --> category")
print("--------------------------------------")

# Step1. Get 부문별 Variables

body = ['CRAGE', 'BMP','SBP_1',
       'SBP_2', 'DBP_1', 'DBP_2',
        'BFP_1', 'BFP_2', 'BMI_1', 'BMI_2']
lung = ['CRAGE','FVCPP_1', 'FVCPP_2',
        'FEV1PP_1', 'FEV1PP_2', 'FEV1PP_3']
blood = ['CRAGE','TOTPRO', 'ALB', 'TOTBIL',
         'ALP', 'LDH', 'CREA', 'BUN',
       'URICACID', 'SODIUM', 'POTASSIUM',
         'CHLORIDE', 'CALCIUM', 'PHOSPHORUS',
       'RBC', 'HCT', 'MCV', 'MCH', 'MCHC',
         'PLAT', 'NEUTROPHIL', 'LYMPHOCYTE', 
         'MONOCYTE', 'EOSINOPHIL', 'BASOPHIL',
         'AST_1','AST_2', 'ALT_1', 'ALT_2', 
         'GGT_1', 'GGT_2', 'CHOL_1', 'CHOL_2',
       'HDL_1', 'HDL_2', 'TG_1', 'TG_2', 
         'GLU_1', 'GLU_2', 'HB_1', 'HB_2']

# res = 갑상선, 면역혈청, 종양표지자
res_m = ['CRAGE','TSH', 'FT4', 'HBSAG', 'RF',
         'ANTIHCVAB', 'CEA', 'AFP', 'CA199', 'PSA',
        'FVCPP_1', 'FVCPP_2',
        'FEV1PP_1', 'FEV1PP_2', 'FEV1PP_3']
res_f = ['CRAGE','TSH', 'FT4', 'HBSAG', 'RF',
         'ANTIHCVAB', 'CEA', 'AFP', 'CA125', 'CA199',
        'FVCPP_1', 'FVCPP_2',
        'FEV1PP_1', 'FEV1PP_2', 'FEV1PP_3']
urine = ['CRAGE','UNITR', 'UPH', 'UPRO', 'UGLU',
         'UKET', 'URO', 'UBIL', 'UBLD', 'USG']


# Step1. Divide by age.
def divide_by_age(bioage_m, bioage_f): # Vars starts at CRAGE
    
    # Male. Divide by age 
    age_m = bioage_m.CRAGE
    G_m1 = [] ; G_m2 = [] ; G_m3 = []; G_m4 = [] ; G_m5 = [] ; G_m6 = []

    for i, age_m in enumerate(age_m):
        if 18 <= age_m and age_m <=30:
            G_m1.append(i)
        if 31 <= age_m and age_m <=40:
            G_m2.append(i)
        if 41 <= age_m and age_m <=50:
            G_m3.append(i)
        if 51 <= age_m and age_m <=60:
            G_m4.append(i)
        if 61 <= age_m and age_m <=70:
            G_m5.append(i)
        if 71 <= age_m:
            G_m6.append(i)
    
    group_m1 = bioage_m.iloc[G_m1,:]
    group_m2 = bioage_m.iloc[G_m2,:]
    group_m3 = bioage_m.iloc[G_m3,:]
    group_m4 = bioage_m.iloc[G_m4,:]
    group_m5 = bioage_m.iloc[G_m5,:]
    group_m6 = bioage_m.iloc[G_m6,:]
    
    print( " ----------------------------------------- ")
    print( " Male : The number of element of each sets ")
    print(len(G_m1),len(G_m2),len(G_m3),len(G_m4),len(G_m5),len(G_m6))
    print( " ----------------------------------------- ")
    
    age_f = bioage_f.CRAGE
    G_f1 = [] ; G_f2 = [] ; G_f3 = []; G_f4 = [] ; G_f5 = [] ; G_f6 = []

    for i, age_f in enumerate(age_f):
        if 18 <= age_f and age_f <=30:
            G_f1.append(i)
        if 31 <= age_f and age_f <=40:
            G_f2.append(i)
        if 41 <= age_f and age_f <=50:
            G_f3.append(i)
        if 51 <= age_f and age_f <=60:
            G_f4.append(i)
        if 61 <= age_f and age_f <=70:
            G_f5.append(i)
        if 71 <= age_f:
            G_f6.append(i)
            
    group_f1 = bioage_f.iloc[G_f1,:]
    group_f2 = bioage_f.iloc[G_f2,:]
    group_f3 = bioage_f.iloc[G_f3,:]
    group_f4 = bioage_f.iloc[G_f4,:]
    group_f5 = bioage_f.iloc[G_f5,:]
    group_f6 = bioage_f.iloc[G_f6,:]
    
    print( " ----------------------------------------- ")
    print( " Female : The number of element of each sets ")
    print(len(G_f1),len(G_f2),len(G_f3),len(G_f4),len(G_f5),len(G_f6))
    print( " ----------------------------------------- ")
    ls = [group_m1, group_m2, group_m3, group_m4, group_m5, group_m6,
          group_f1, group_f2, group_f3, group_f4, group_f5, group_f6]
    return ls


groups_age = divide_by_age(bioage_m = bioage_m, bioage_f = bioage_f)
print( "----------------------------- ")
print("Division by age is finished")
print( "----------------------------- ")
time.sleep(3)



# Step2. Divide data into variables
def divide_by_age_var_m(group):
    group_body = group[body]
    group_lung = group[lung]
    group_blood = group.loc[:,blood]
    group_res = group.loc[:,res_m]
    group_urine = group.loc[:, urine]
    ls = [group_body,group_lung,group_blood,group_res,group_urine]
    return ls

def divide_by_age_var_f(group):
    group_body = group.loc[:,body]
    group_lung = group.loc[:,lung]
    group_blood = group.loc[:,blood]
    group_res = group.loc[:,res_f]
    group_urine = group.loc[:, urine]
    ls = [group_body,group_lung,group_blood,group_res,group_urine]
    return ls


def divide_by_age_var(groups):
    dic = {}
    for idx, group in enumerate(groups):
        if idx <= 5:
            name = "group_m_%s" % idx
            group_m_var = divide_by_age_var_m(group=group)
            dic[name] = group_m_var
        
        if idx > 5:
            name = "group_f_%s" % (idx-6)
            group_f_var = divide_by_age_var_f(group=group)
            dic[name] = group_f_var
    return dic

groups_age_var = divide_by_age_var(groups = groups_age)
print("Division by age and variable is finished")
time.sleep(3)

outfile = open("data/bioage_change_divide",'wb')
pickle.dump(groups_age_var,outfile)
outfile.close()
print("Save it as bioage_change_divide")
