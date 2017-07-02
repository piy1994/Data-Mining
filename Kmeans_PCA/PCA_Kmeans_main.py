import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import math
import sys

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
        #print('iterations')      
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
    #print('done making matrix')
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
 
 
def kmeans_with_results(df_path,k):
    dig_df = pd.read_csv(df_path,header= None,names = ['id','label','dim1','dim2'])
    img_vec = np.array(dig_df.drop(['id','label'],axis=1)).astype(np.float)
    orig_labels = np.array(dig_df['label'])
    centroid_orig = original_centroid(img_vec,orig_labels)
    
    centroid_idx,centroid = kmeans(img_vec,k)
    dist = w_ssd(centroid_idx,img_vec,k,centroid)
    sil = silhoutte_coeff(centroid_idx,img_vec)
    nmi_val = NMI_fnc(orig_labels,centroid_idx)
    return dist,sil,nmi_val
    
def main():
    #print('hello')
    num_imputs = len(sys.argv[1:])
    inputs = sys.argv[1:]
    #print(num_imputs)
    df_path = inputs[0]
    k = int(inputs[1])

    dist,sil,nmi_val = kmeans_with_results(df_path,k)
    print 'WC-SSD {} '.format(dist)
    print 'SC {} '.format(sil)
    print 'NMI {} '.format(nmi_val)
 
if __name__ == "__main__":
    main()  
    
#%% PART C
#import numpy as np
#import pandas as pd
#from scipy.cluster import hierarchy
#import matplotlib.pyplot as plt
#from kmeans_fnc import *

#
#d_embed = pd.read_csv('digits-embedding.csv',header = None)
##lst_subsample = []
#num_sample = 2000
#num_digits = 10
#
#sub_sample_label = np.empty((0,1))
#sub_sample = np.empty((0,2))
#for i in range(num_digits):
#    temp_sample = np.array(d_embed[d_embed[1] == i][[1,2,3]].sample(10))
#    #lst_subsample.append(temp_sample[:,1:])
#    sub_sample = np.vstack((sub_sample,temp_sample[:,1:]))
#    sub_sample_label = np.vstack((sub_sample_label,
#                                  temp_sample[:,0].reshape((len(temp_sample[:,0]),1))))
#del i,temp_sample
##%%
#
#single_link = hierarchy.linkage(sub_sample, method='single', metric='euclidean')
##single_dendo = hierarchy.dendrogram(single_link)
#
#avg_link  = hierarchy.linkage(sub_sample, method='average', metric='euclidean')
##avg_dendo = hierarchy.dendrogram(avg_link)
#
#comp_link  = hierarchy.linkage(sub_sample, method='complete', metric='euclidean')
##comp_dendo = hierarchy.dendrogram(comp_link)
#
##a = hierarchy.fcluster(single_link, 10, criterion='maxclust')
##plt.scatter(sub_sample[:,0],sub_sample[:,1],c = a)
##del comp_dendo,avg_dendo,single_dendo
##%% plots
#lst_linkages = [single_link,avg_link,comp_link]
#orig_labels = sub_sample_label[:,0].tolist()
##%%
#k_lst = [2,4,8,16,32]
#
#wssd_link = []
#sil_link = []
#nmi_link = [] 
#for i in range(3):
#    link = lst_linkages[i]
#    w_ssd_lst = []
#    sil_lst = []
#    nmi_lst = []
#    
#    for k in k_lst:
#        labels = hierarchy.fcluster(link, k, criterion='maxclust')
#        labels = labels-1
#        centroids = original_centroid(sub_sample,labels)
#        dist = w_ssd(labels,sub_sample,k,centroids)
#        sil = silhoutte_coeff(labels,sub_sample)
#        #sil = silhouette_score(sub_sample,labels)
#        nmi_val = NMI_fnc(orig_labels,labels)
#        w_ssd_lst.append(dist)
#        sil_lst.append(sil)
#        nmi_lst.append(nmi_val)
#    wssd_link.append(w_ssd_lst)
#    sil_link.append(sil_lst)
#    nmi_link.append(nmi_lst)
##%%
#plt.plot(k_lst, sil_link[0],label = 'Single')
#plt.plot(k_lst, sil_link[1],label = 'Average')
#plt.plot(k_lst, sil_link[2],label = 'Complete')
#plt.title('sil_link for 3 linkages with K')
##plt.legend()
#
# 
##%%
#%% PCA

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#def norm_image(img_vec):
#    for i in range(img_vec.shape[1]):
#        img_vec[:,i] = img_vec[:,i] - img_vec[:,i].mean()
#    return img_vec
#
#def pca(df,k):
#    
#    dig = df.copy()
#    img_vec = np.array(dig.drop([0,1],axis=1)).astype(np.float)
#    img_vec = img_vec.T
#    img_vec_mean = norm_image(img_vec.copy())
#    img_cov = np.cov(img_vec_mean)
#    U,_,_ = np.linalg.svd(img_cov)
#    U_red = U[:,:k]
#    img_vec_red = np.dot(U_red.T,img_vec_mean)
#    img_vec_for_repre = np.dot(U[:,:2].T,img_vec_mean)
#    return img_vec.T,img_vec_red.T,U_red,img_vec_for_repre.T
#
#df = pd.read_csv('digits-raw.csv',header = None)
#img_vec,img_vec_red_mean,U_red,img_vec_for_repre = pca(df,k=10)
#
#idx = np.random.choice(np.array(df[0]), size=1000, replace=False)
#plt.scatter(img_vec_for_repre[idx,0],img_vec_for_repre[idx,1],c = np.array(df[1].loc[idx]))
##str_ = 'E:\study_course Purdue\datamining\hw5\update_2\results\pca_images'
#
##for i in range(10):
##    a = U_red[:,i].reshape((28,28))
##    plt.imshow(a,cmap=plt.gray())
##    plt.title('Principal Component {}'.format(i))
##    plt.savefig('Component_{}.jpg'.format(i))
#
##U_red_dim2 = U_red[:,:2]
##img_vec_red_dim2 = np.dot(U_red_dim2.T,img_vec.T).T
##df_dim2  = pd.DataFrame(img_vec_red_dim2,columns = ['dim1','dim2'])
##
##df_dim2.insert(0,'label',df[1])
##df_dim2.insert(0,'id',df[0])
##df_dim2.to_csv('PCA_embedding.csv',index = False)
#
#
#df_dim10 = pd.DataFrame(img_vec_red_mean,columns = ['dim1','dim2','dim3','dim4',
#                                                    'dim5','dim6','dim7','dim8',
#                                                    'dim9','dim10'])
#df_dim10.insert(0,'label',df[1])
#df_dim10.insert(0,'id',df[0])
#df_dim10.to_csv('PCA_embedding.csv',index = False)

#%%
    