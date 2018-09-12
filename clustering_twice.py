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



# Bauckhage C. Numpy/scipy Recipes for Data Science:
# k-Medoids Clustering[R]. Technical Report, University of Bonn, 2015.
def kMedoids(D, k, seed= 2525,tmax=100): # D = pairwise distance = [[1stobs와 나머지 거리],[2nd obs와 나머지 거리], ..., ]
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.seed(seed)
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C


def get_the_score(group1, group2, kind):
    
    if kind == "urine":
        
        idx_score_sum1 = 0
        idx_score_sum2 = 0
        
        for idx in group1.index:
            tmp_score = 0
            for i, var in enumerate(urine_categorical):
                if group1.loc[idx, var] == "Positive":
                    tmp_score = tmp_score + 1
                elif group1.loc[idx, var] == "Trace":
                    tmp_score = tmp_score + 0.5
            idx_score_sum1 = idx_score_sum1 + tmp_score
        idx_score_sum1 = idx_score_sum1/group1.shape[0]
            
        for idx in group2.index:
            tmp_score = 0
            for i, var in enumerate(urine_categorical):
                if group2.loc[idx, var] == "Positive":
                    tmp_score = tmp_score + 1
                elif group2.loc[idx, var] == "Trace":
                    tmp_score = tmp_score + 0.5
            idx_score_sum2 = idx_score_sum2 + tmp_score
        idx_score_sum2 = idx_score_sum2/group2.shape[0]
        
    # Get 
    elif kind == "body":
        categorical_score = body_categorical_score
        categorical_vars = body_categorical
    elif kind == "lung":
        categorical_score = lung_categorical_score
        categorical_vars = lung_categorical
    elif kind == "blood":
        categorical_score = blood_categorical_score
        categorical_vars = blood_categorical
    elif kind == "res":
        categorical_score = res_categorical_score
        categorical_vars = res_categorical
    
    if kind != "urine":
        
        idx_score_sum1 = 0
        idx_score_sum2 = 0
        
        for idx in group1.index:
            tmp_score = 0
            for i, var in enumerate(categorical_vars):
                if group1.loc[idx, var] == "Positive":
                    tmp_score = tmp_score + categorical_score[i]
            idx_score_sum1 = idx_score_sum1 + tmp_score
        idx_score_sum1 = idx_score_sum1/group1.shape[0]
    
        for idx in group2.index:
            tmp_score = 0
            for i, var in enumerate(categorical_vars):
                if group2.loc[idx, var] == "Positive":
                    tmp_score = tmp_score + categorical_score[i]
            idx_score_sum2 = idx_score_sum2 + tmp_score
        idx_score_sum2 = idx_score_sum2/group2.shape[0]
        
    return idx_score_sum1, idx_score_sum2


def get_ordered_clusters(group, kind, method="score"):
    
    print("Group size")
    print(group.shape[0])
    # time.sleep(0.5)
    
    # step1. Get gower distance
    group_without_age = group.iloc[:,1:]
    D = gower_distance(group_without_age)
    
    # step2. Get 2Medoids, C is not index but 0부터의 순서, 
    M, C = kMedoids(D, 2)
    G_1 = group.iloc[C[0],:]
    G_2 = group.iloc[C[1],:]
    

    # step3. Get mean of each group
    if method == "age":
        A_1 = G_1.CRAGE.mean()
        A_2 = G_2.CRAGE.mean()
    
    # Step4. Get the score
    if method == "score":
        A_1, A_2 = get_the_score(group1 = G_1, group2 = G_2, kind = kind)
    
    # Step5. order two groups
    if A_1 <= A_2:
        group_1 = G_1
        group_2 = G_2
        score_1 = A_1
        score_2 = A_2
    else:
        group_2 = G_1
        group_1 = G_2
        score_1 = A_2
        score_2 = A_1
        
    return [group_1, group_2, score_1, score_2]



# ---------------------------------------------------------------
infile = open("data/bioage_change_divided_group1_compare1",'rb')
tmp = pickle.load(infile)
group_m_4, group_m_5, group_f_4, group_f_5 = tmp[0],tmp[1],tmp[2],tmp[3]
kind = ["body","body", "lung","lung", "blood", "blood", "res", "res", "urine", "urine"]



# Step1. Find categorical variables

group_m_4_body = group_m_4[0]
group_m_4_lung = group_m_4[2]
group_m_4_blood = group_m_4[4]
group_m_4_res = group_m_4[6]
group_m_4_urine = group_m_4[8]

ls = []
for j in range(group_m_4_body.shape[1]):
        if group_m_4_body.dtypes[j] == object or group_m_4_body.dtypes[j] == np.object:
              ls.append(j)
body_categorical = group_m_4_body.columns[ls]

ls = []
for j in range(group_m_4_lung.shape[1]):
        if group_m_4_lung.dtypes[j] == object or group_m_4_lung.dtypes[j] == np.object:
              ls.append(j)
lung_categorical = group_m_4_lung.columns[ls]

ls = []
for j in range(group_m_4_blood.shape[1]):
        if group_m_4_blood.dtypes[j] == object or group_m_4_blood.dtypes[j] == np.object:
              ls.append(j)
blood_categorical = group_m_4_blood.columns[ls]

ls = []
for j in range(group_m_4_res.shape[1]):
        if group_m_4_res.dtypes[j] == object or group_m_4_res.dtypes[j] == np.object:
              ls.append(j)
res_categorical = group_m_4_res.columns[ls]

ls = []
for j in range(group_m_4_urine.shape[1]):
        if group_m_4_urine.dtypes[j] == object or group_m_4_urine.dtypes[j] == np.object:
              ls.append(j)
urine_categorical = group_m_4_urine.columns[ls]


# Step2. Set the score of it
body_categorical_score = [0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1]
lung_categorical_score = [0.5, 1, 1, 1, 2]
blood_categorical_score = [0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1]
#res_categorical_score = [1, 1, 1]
res_categorical_score = [1, 1, 1, 0.5, 1, 1, 1, 2]
# Urine --> trace = 0.5, Positive = 1


group_divided_once_m_4 = []
score_m_4 = []
for i, group_m_4_var in enumerate(group_m_4):
    print( "----------------------------------")
    print( "----------------------------------")
    print( "group_m_4_" + kind[i] + " is started" )
    group1, group2, score1, score2 = get_ordered_clusters(group_m_4_var, kind = kind[i], method = 'score')
    group_divided_once_m_4.append(group1)
    group_divided_once_m_4.append(group2)
    score_m_4.append(score1)
    score_m_4.append(score2)
    print( "Younger Group Score : ", score1)
    print( "Older Group Score : ", score2 )
    print( "Done" )
    # time.sleep(1)
print( "Done : group_m_4" )
print( "Yeah~~~~~~~~~~~" )
print( "----------------------------------")
print( "----------------------------------")

group_divided_once_m_5 = []
score_m_5 = []
for i, group_m_5_var in enumerate(group_m_5):
    print( "----------------------------------")
    print( "----------------------------------")
    print( "group_m_5_" + kind[i] + " is started" )
    group1, group2, score1, score2 = get_ordered_clusters(group_m_5_var,  kind = kind[i], method = "score")
    group_divided_once_m_5.append(group1)
    group_divided_once_m_5.append(group2)
    score_m_5.append(score1)
    score_m_5.append(score2)
    print( "Younger Group Score : ", score1)
    print( "Older Group Score : ", score2 )
    print( "Done" )
    # time.sleep(1)
print( "Done : group_m_5" )
print( "Yeah~~~~~~~~~~~" )
print( "----------------------------------")
print( "----------------------------------")

tmp = [group_divided_once_m_4, group_divided_once_m_5]
outfile = open("data/bioage_m_change_divided_group1_compare1_group2",'wb')
pickle.dump(tmp,outfile)
outfile.close()
       
tmp_score = [score_m_4, score_m_5]
outfile_score = open("data/clustering_twice_m_score",'wb')
pickle.dump(tmp_score,outfile_score)
outfile_score.close()


group_divided_once_f_4 = []
score_f_4 = []
for i, group_f_4_var in enumerate(group_f_4):
    print( "----------------------------------")
    print( "----------------------------------")
    print( "group_f_4_" + kind[i] + " is started" )
    group1, group2, score1, score2 = get_ordered_clusters(group_f_4_var,  kind = kind[i], method = "score")
    group_divided_once_f_4.append(group1)
    group_divided_once_f_4.append(group2)
    score_f_4.append(score1)
    score_f_4.append(score2)
    print( "Younger Group Score : ", score1)
    print( "Older Group Score : ", score2 )
    print( "Done" )
    # time.sleep(1)
print( "Done : group_f_4" )
print( "Yeah~~~~~~~~~~~" )
print( "----------------------------------")
print( "----------------------------------")

group_divided_once_f_5 = []
score_f_5 = []
for i, group_f_5_var in enumerate(group_f_5):
    print( "----------------------------------")
    print( "----------------------------------")
    print( "group_f_5_" + kind[i] + " is started" )
    group1, group2, score1, score2 = get_ordered_clusters(group_f_5_var,  kind = kind[i], method = "score")
    group_divided_once_f_5.append(group1)
    group_divided_once_f_5.append(group2)
    score_f_5.append(score1)
    score_f_5.append(score2)
    print( "Younger Group Score : ", score1)
    print( "Older Group Score : ", score2 )
    print( "Done" )
    # time.sleep(1)
print( "Done : group_f_5" )
print( "Yeah~~~~~~~~~~~" )
print( "----------------------------------")
print( "----------------------------------")

tmp2 = [group_divided_once_f_4, group_divided_once_f_5]
outfile2 = open("data/bioage_f_change_divided_group1_compare1_group2",'wb')
pickle.dump(tmp2,outfile2)
outfile2.close()


tmp3 = [score_f_4, score_f_5]
outfile3 = open("data/clustering_twice_f_score",'wb')
pickle.dump(tmp3, outfile3)
outfile3.close()





