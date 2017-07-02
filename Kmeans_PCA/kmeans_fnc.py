import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math

def get_centroid(dataset,lst_centroid):
    main_dist = np.empty((0,len(dataset)))
    lst_centroid_new = np.zeros(lst_centroid.shape)
    for i in range(len(lst_centroid)):
        dist = np.linalg.norm(dataset-lst_centroid[i],axis=1)
        main_dist = np.vstack((main_dist,dist))
    centroid_idx = np.argmin(main_dist,axis=0)
    for i in range(len(lst_centroid)):
        #print(dataset[centroid_idx == i])
        #print(np.mean(dataset[centroid_idx == i],axis=0))
        lst_centroid_new[i] = np.mean(dataset[centroid_idx == i],axis=0)
    return centroid_idx,lst_centroid_new
    

def shouldStop(oldCentroids, centroids, iterations,max_iterations = 50):
    if iterations > max_iterations:
        print('iterations')      
        return True
    return np.array_equal(oldCentroids,centroids)
    
def get_random_centroid(dataset,k):
    return dataset[np.random.choice(dataset.shape[0], k, replace=False), :]
    

def kmeans(dataset,k):
    centroid = get_random_centroid(dataset.copy(),k)
    #print(centroid)
    max_iterations = 50
#    Initialize book keeping vars.
    iterations = 0
    oldCentroids = None
    
    while not shouldStop(oldCentroids,centroid,iterations,max_iterations):
        oldCentroids = centroid.copy()
        iterations+=1
        centroid_idx,centroid = get_centroid(dataset.copy(),centroid.copy())
    return centroid_idx,centroid

def w_ssd(centroid_idx,dataset,k,centroid):
    dist = 0
    num_dim = dataset.shape[1]
    for i in range(k):
        #dist += np.linalg.norm(dataset[centroid_idx == i,:] - centroid[i])
        d_temp = dataset[centroid_idx == i,:]
        dist +=  cdist(d_temp,centroid[i].reshape((1,num_dim)),'sqeuclidean').sum()
    return dist

def original_centroid(dataset,orig_labels):
    unique_labels = np.unique(orig_labels)
    lst_centroid_orig = np.zeros((len(unique_labels),dataset.shape[1]))
    for i in unique_labels:
        lst_centroid_orig[i] = np.mean(dataset[orig_labels == i],axis=0)
    return lst_centroid_orig
    


def intra_cluster_dist(dist_mat_row,centroid_idx,label):
    return dist_mat_row[centroid_idx == label].mean()

def nearest_cluster_dist(dist_mat_row,centroid_idx,near_labels):
    lst_dist = []
    for label in near_labels:
        dist_temp = dist_mat_row[centroid_idx == label].mean()
        lst_dist.append(dist_temp)
    return min(lst_dist)



def silhoutte_coeff(centroid_idx,dataset):
    s = []
    u  = (np.unique(centroid_idx))

    dist_mat = cdist(dataset,dataset,metric='euclidean')
    print('done making matrix')
    for sam_idx in range(dataset.shape[0]):
        a = intra_cluster_dist(dist_mat[sam_idx,:],
                               centroid_idx,centroid_idx[sam_idx])
                    
        b = nearest_cluster_dist(dist_mat[sam_idx,:],centroid_idx,
                                 u[u!=centroid_idx[sam_idx]])
        s.append((b-a)/max(a,b))
    return np.array(s).mean()
    
def p_times_log(prob,pc,pg):
    if prob==0:
        return 0
    else:
        return prob*math.log((prob)/(pc*pg))
    

def NMI_fnc(orig_label,centroid_idx):
    g = np.unique(centroid_idx)
    c = np.unique(orig_label)

    prob_mat = np.zeros((len(c),len(g)))
    for cls in c:
        for grp in g:
            prob_mat[cls,grp] = np.sum((orig_label==cls) & (centroid_idx == grp))

    prob_mat = prob_mat/len(orig_label)

    p_c = np.sum(prob_mat,axis=1)
    p_g = np.sum(prob_mat,axis=0)

    num=0
    for cls in c:
        for grp in g:
            num+= p_times_log(prob_mat[cls,grp],p_c[cls],p_g[grp])

    denom_c = 0
    for cls in c:
        denom_c+=p_c[cls]*math.log(p_c[cls])
    
    denom_g=0
    for grp in g:
        denom_g+= p_g[grp]*math.log(p_g[grp])
    
    NMI = num/(-0.5*(denom_c+denom_g))
    return NMI
 