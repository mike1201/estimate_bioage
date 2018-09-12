import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import sklearn.cluster as sc
import time
import functions
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

def gower_distance(X):
    individual_variable_distances = []
    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
        individual_variable_distances.append(feature_dist)
        
    return np.array(individual_variable_distances).mean(0)

def get_KNN(pair_distance, k): # pairwise distance = [[1st obs와 나머지 거리],[2nd obs와 나머지 거리], ..., ]
    ls = []
    l = k-1
    for dis in pair_distance:
        order = dis.argsort().argsort()
        ls2 = []
        for idx, ord in enumerate(order):
            if ord <= l:
                ls2.append(idx)
        ls.append(ls2)
    return ls


def compare_near_group(group1, group2, n_knn=20, back = False):
    
    # step1. Concatenate groups
    frames = [group1, group2]
    group_concat = pd.concat(frames,0)
    
    # 이렇게 바꿔주면 gower_distance : 어차피 column으로 loop --> pairwise distance
    # get_KNN --> pair_distance로 loop --> 순서바뀐 것이 반영됨.
    #if back == True:
    #    group_concat = group_concat.reindex(index=group_concat.index[::-1])
    #    group1 = group1.reindex(index= group1.index[::-1])
    #    group2 = group2.reindex(index= group2.index[::-1])
    
    # step2. Get KNN for each points.
    group_concat_without_age = group_concat.iloc[:,1:]
    pairwise_distance = gower_distance(group_concat_without_age)
    KNN = get_KNN(pairwise_distance, k=n_knn)
    
    # step3. Get the number of KNN which belongs to another group.
    dic1 = {}
    dic2 = {}    
    
    for i, k in enumerate(KNN):
        
        # Get index of data and KNN_neighbor
        count = 0
        data_index = group_concat.index[i]
        neighbor_index = group_concat.index[k]
        
        for idx in neighbor_index:
            if data_index in group1.index:
                if idx in group2.index:
                    count = count + 1
                dic1[data_index] = count
            if data_index in group2.index:
                if idx in group1.index:
                    count = count + 1
                dic2[data_index] = count 
                
    # step4. Move the data considering group size
    size1 = group1.shape[0]
    size2 = group2.shape[0]
    prop1 = size1 / (size1 + size2)
    prop2 = size2 / (size1 + size2)
    print("-----------")
    print("proportion is")
    print(prop1, prop2)
    # time.sleep(1)
    print("-----------")
    print("group size is")
    print(size1, size2)
    #time.sleep(1)
    
    ls1 = list(group1.index)
    ls2 = list(group2.index)
    count1 = 0
    count2 = 0
    
    if prop1 > 0.9:
        for idx, count in dic1.items():
            if count > n_knn*(prop2+0.1):
                ls1.remove(idx)
                ls2.append(idx)
                count1 = count1 + 1
                
    elif prop2 > 0.9:
        for idx, count in dic2.items():
            if count > n_knn*(prop1+0.1):
                ls2.remove(idx)
                ls1.append(idx)
                count2 = count2 + 1
                
    else:
        for idx, count in dic1.items():
            if count > n_knn*(prop2-0.15):
                ls1.remove(idx)
                ls2.append(idx)
                count1 = count1 + 1
        for idx, count in dic2.items():
            if count > n_knn*(prop1-0.15):
                ls2.remove(idx)
                ls1.append(idx)
                count2 = count2 + 1
    '''
    for idx, count in dic1.items():
        if count > n_knn*(prop2-0.03):
            ls1.remove(idx)
            ls2.append(idx)
            count1 = count1 + 1
    for idx, count in dic2.items():
        if count > n_knn*(prop1):
            ls2.remove(idx)
            ls1.append(idx)
            count2 = count2 + 1
    '''
    
    group1 = group_concat.loc[ls1,:]
    group2 = group_concat.loc[ls2,:]
    return [group1, group2, count1, count2, prop1, prop2]


# ---- Comparison -------

infile = open("data/bioage_m_change_divide_groupscore",'rb')
group_divided_once_m = pickle.load(infile)
infile2 = open("data/bioage_f_change_divide_groupscore",'rb')
group_divided_once_f = pickle.load(infile2)

# Step1. Set groups to compare

group_divided_once_m_4 = group_divided_once_m[0]
group_divided_once_m_5 = group_divided_once_m[1]
group_divided_once_f_4 = group_divided_once_f[0]
group_divided_once_f_5 = group_divided_once_f[1]

m_4_body_1 = group_divided_once_m_4[0]; m_4_body_2 = group_divided_once_m_4[1]; m_4_lung_1 = group_divided_once_m_4[2]
m_4_lung_2 = group_divided_once_m_4[3]; m_4_blood_1 = group_divided_once_m_4[4]; m_4_blood_2 = group_divided_once_m_4[5]
m_4_res_1 = group_divided_once_m_4[6]; m_4_res_2 = group_divided_once_m_4[7]; m_4_urine_1 = group_divided_once_m_4[8]
m_4_urine_2 = group_divided_once_m_4[9]

m_5_body_1 = group_divided_once_m_5[0]; m_5_body_2 = group_divided_once_m_5[1]; m_5_lung_1 = group_divided_once_m_5[2]
m_5_lung_2 = group_divided_once_m_5[3]; m_5_blood_1 = group_divided_once_m_5[4]; m_5_blood_2 = group_divided_once_m_5[5]
m_5_res_1 = group_divided_once_m_5[6]; m_5_res_2 = group_divided_once_m_5[7]; m_5_urine_1 = group_divided_once_m_5[8]
m_5_urine_2 = group_divided_once_m_5[9]

f_4_body_1 = group_divided_once_f_4[0]; f_4_body_2 = group_divided_once_f_4[1]; f_4_lung_1 = group_divided_once_f_4[2]
f_4_lung_2 = group_divided_once_f_4[3]; f_4_blood_1 = group_divided_once_f_4[4]; f_4_blood_2 = group_divided_once_f_4[5]
f_4_res_1 = group_divided_once_f_4[6]; f_4_res_2 = group_divided_once_f_4[7]; f_4_urine_1 = group_divided_once_f_4[8]
f_4_urine_2 = group_divided_once_f_4[9]

f_5_body_1 = group_divided_once_f_5[0]; f_5_body_2 = group_divided_once_f_5[1]; f_5_lung_1 = group_divided_once_f_5[2]
f_5_lung_2 = group_divided_once_f_5[3]; f_5_blood_1 = group_divided_once_f_5[4]; f_5_blood_2 = group_divided_once_f_5[5]
f_5_res_1 = group_divided_once_f_5[6]; f_5_res_2 = group_divided_once_f_5[7]; f_5_urine_1 = group_divided_once_f_5[8]
f_5_urine_2 = group_divided_once_f_5[9]


compare_groups = [[m_4_body_2, m_5_body_1], [m_4_lung_2, m_5_lung_1], [m_4_blood_2, m_5_blood_1], 
                  [m_4_res_2, m_5_res_1], [m_4_urine_2, m_5_urine_1],[f_4_body_2, f_5_body_1], 
                  [f_4_lung_2, f_5_lung_1], [f_4_blood_2, f_5_blood_1],[f_4_res_2, f_5_res_1], [f_4_urine_2, f_5_urine_1]]

kind = ["m_body", "m_lung", "m_blood", "m_res", "m_urine", "f_body", "f_lung", "f_blood", "f_res", "f_urine"]
kind2 = ["body","lung","blood","res","urine","body","lung","blood","res","urine"]


# Step2. Compare near groups many times

print( "Update V1 : Started" )
print( "Update V1 : Started" )
print( "Update V1 : Started" )
compare_groups_V1 = []
counts_V1 = []
props_V1 = []
for i, compare_group in enumerate(compare_groups):
    group1 = compare_group[0]
    group2 = compare_group[1]
    groups = compare_near_group(group1 = group1, group2 = group2, n_knn=20)
    group_1 = groups[0]; group_2 = groups[1]; count_1 = groups[2]; count_2 = groups[3]; prop_1 = groups[4]; prop_2 = groups[5]
    print(kind[i])
    print("Young --> Old : ", count_1, "  Old --> Young : ", count_2  )
    print("---------------------------------------------------------")
    # time.sleep(1)
    compare_groups_V1.append([group_1, group_2])
    counts_V1.append([count_1, count_2]) 
    props_V1.append([prop_1, prop_2])
    
print(props_V1)
    
print( "Update V2 : Started" )
print( "Update V2 : Started" )
print( "Update V2 : Started" )
compare_groups_V2 = []
counts_V2 = []
props_V2 = []
for i, compare_group in enumerate(compare_groups_V1):
    group1 = compare_group[0]
    group2 = compare_group[1]
    groups = compare_near_group(group1 = group1, group2 = group2, n_knn=20, back = True)
    group_1 = groups[0]; group_2 = groups[1]; count_1 = groups[2]; count_2 = groups[3]; prop_1 = groups[4]; prop_2 = groups[5]
    print(kind[i])
    print("Young --> Old : ", count_1, "  Old --> Young : ", count_2  )
    print("---------------------------------------------------------")
    # time.sleep(0.5)
    compare_groups_V2.append([group_1, group_2])
    counts_V2.append([count_1, count_2])
    props_V2.append([prop_1, prop_2])
print(props_V2)

print( "Update V3 : Started" )
print( "Update V3 : Started" )
print( "Update V3 : Started" )
compare_groups_V3 = []
counts_V3 = []
props_V3 = []
for i,compare_group in enumerate(compare_groups_V2):
    group1 = compare_group[0]
    group2 = compare_group[1]
    groups = compare_near_group(group1 = group1, group2 = group2, n_knn=20)
    group_1 = groups[0]; group_2 = groups[1]; count_1 = groups[2]; count_2 = groups[3]; prop_1 = groups[4]; prop_2 = groups[5]
    print(kind[i])
    print("Young --> Old : ", count_1, "  Old --> Young : ", count_2  )
    print("---------------------------------------------------------")
    # time.sleep(0.5)
    compare_groups_V3.append([group_1, group_2])
    counts_V3.append([count_1, count_2])
    props_V3.append([prop_1, prop_2])
    
print( "Update V4 : Started" )
print( "Update V4 : Started" )
print( "Update V4 : Started" )
compare_groups_V4 = []
counts_V4 = []
props_V4 = []
for i,compare_group in enumerate(compare_groups_V3):
    group1 = compare_group[0]
    group2 = compare_group[1]
    groups = compare_near_group(group1 = group1, group2 = group2, n_knn=20, back = True)
    group_1 = groups[0]; group_2 = groups[1]; count_1 = groups[2]; count_2 = groups[3]; prop_1 = groups[4]; prop_2 = groups[5]
    print(kind[i])
    print("Young --> Old : ", count_1, "  Old --> Young : ", count_2  )
    print("---------------------------------------------------------")
    # time.sleep(0.5)
    compare_groups_V4.append(group_1)
    compare_groups_V4.append(group_2)
    counts_V4.append([count_1, count_2])
    props_V4.append([prop_1, prop_2])

# Step3. Get result and Save it
m_4_body_2, m_5_body_1, m_4_lung_2, m_5_lung_1, m_4_blood_2, m_5_blood_1, m_4_res_2, m_5_res_1, m_4_urine_2, m_5_urine_1,f_4_body_2, f_5_body_1, f_4_lung_2, f_5_lung_1, f_4_blood_2, f_5_blood_1,f_4_res_2, f_5_res_1,f_4_urine_2, f_5_urine_1 = compare_groups_V4[0],compare_groups_V4[1],compare_groups_V4[2],compare_groups_V4[3],compare_groups_V4[4],compare_groups_V4[5],compare_groups_V4[6],compare_groups_V4[7],compare_groups_V4[8],compare_groups_V4[9],compare_groups_V4[10],compare_groups_V4[11],compare_groups_V4[12],compare_groups_V4[13],compare_groups_V4[14],compare_groups_V4[15],compare_groups_V4[16],compare_groups_V4[17],compare_groups_V4[18],compare_groups_V4[19]

# 

group_m_4 = [m_4_body_1, m_4_body_2, m_4_lung_1, m_4_lung_2, m_4_blood_1, m_4_blood_2, m_4_res_1, m_4_res_2, m_4_urine_1, m_4_urine_2]
group_f_4 = [f_4_body_1, f_4_body_2, f_4_lung_1, f_4_lung_2, f_4_blood_1, f_4_blood_2, f_4_res_1, f_4_res_2, f_4_urine_1, f_4_urine_2]
group_m_5 = [m_5_body_1, m_5_body_2, m_5_lung_1, m_5_lung_2, m_5_blood_1, m_5_blood_2, m_5_res_1, m_5_res_2, m_5_urine_1, m_5_urine_2]
group_f_5 = [f_5_body_1, f_5_body_2, f_5_lung_1, f_5_lung_2, f_5_blood_1, f_5_blood_2, f_5_res_1, f_5_res_2, f_5_urine_1, f_5_urine_2]

divided_once_compare_complete = [group_m_4, group_m_5, group_f_4, group_f_5]
outfile = open("data/bioage_change_divided_group1_compare1",'wb')
pickle.dump(divided_once_compare_complete, outfile)
outfile.close()

counts = [counts_V1, counts_V2, counts_V3, counts_V4, props_V1, props_V2, props_V3, props_V4]
outfile2 = open("data/compare_count_prop",'wb')
pickle.dump(counts, outfile2)
outfile2.close()  

print(props_V4)











