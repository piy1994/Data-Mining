import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def norm_image(img_vec):
    for i in range(img_vec.shape[1]):
        img_vec[:,i] = img_vec[:,i] - img_vec[:,i].mean()
    return img_vec

def pca(df,k):
    
    dig = df.copy()
    img_vec = np.array(dig.drop([0,1],axis=1)).astype(np.float)
    img_vec = img_vec.T
    img_vec_mean = norm_image(img_vec.copy())
    img_cov = np.cov(img_vec_mean)
    U,_,_ = np.linalg.svd(img_cov)
    U_red = U[:,:k]
    img_vec_red = np.dot(U_red.T,img_vec_mean)
    img_vec_for_repre = np.dot(U[:,:2].T,img_vec_mean)
    return img_vec.T,img_vec_red.T,U_red,img_vec_for_repre.T

df = pd.read_csv('digits-raw.csv',header = None)
img_vec,img_vec_red_mean,U_red,img_vec_for_repre = pca(df,k=10)
