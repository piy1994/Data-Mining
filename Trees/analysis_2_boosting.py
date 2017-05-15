import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from pre_proc import *
from decision_tree_class import *
from random_forest import *
from boost_2 import *
from bagging import *
from sv1 import *

df = pd.read_csv('yelp_data.csv', delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                       engine='python')
df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
lst_df = np.array_split(df,10)
#data_sz = [0.025, 0.05, 0.125, 0.25]
#train_size = np.multiply(2000,np.array(data_sz))

function_to_apply = 1
num_feat = 2


err_rand_arr = []
err_decision_arr = []
err_svm_arr = []
err_boost_arr = []
err_bag_arr = []

w_num_features =  [200,500,1000,1500]

max_depth = 10
min_size = 10
num_trees =50
tr_sz = 500
for w in w_num_features:
    err_rand=[]
    err_decision = []
    err_boost = []
    err_svm = []
    err_bag = []
    for i in range(10):
        train_df_mega = pd.concat(lst_df[:i]+lst_df[i+1:]).reset_index(drop=True)
        df_test = lst_df[i].reset_index(drop=True)
        df_train = train_df_mega.sample(tr_sz).reset_index(drop=True)
        
        
        df_train.reviewText = df_train.reviewText.apply(fnc)
        df_temp_wrds_slt = df_train.copy()
        
        
        df_train.reviewText = df_train.reviewText.apply(fnc_vectorize(function_to_apply))
        df_temp_wrds_slt.reviewText = df_temp_wrds_slt.reviewText.apply(fnc_vectorize(1))
        
        
        df_test.reviewText = df_test.reviewText.apply(fnc)
        df_test.reviewText = df_test.reviewText.apply(fnc_vectorize(function_to_apply))
        
        wrds_slt = frequent_words(df_temp_wrds_slt, w)
        df_feat_train = bag_of_feature(df_train, wrds_slt,add_class = True)
        df_feat_test = bag_of_feature(df_test, wrds_slt,add_class = True) 
        
        X_train,Y_train,X_test,Y_test,XY_train,XY_test = dataframe_to_numpy(df_feat_train,df_feat_test)
        
        print('decision')
        err_decision.append(decision_fnc(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size))
        
        print('rand')
        
        err_rand.append(random_fnc(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size,num_trees=num_trees))
        
        print('boost')
        err_boost.append(boosting_function(X_train.copy(),Y_train.copy(),
                                               X_test.copy(),Y_test.copy(),
                                               max_depth=max_depth,
                                               min_size=min_size,num_trees=num_trees))

        print('bag')
        err_bag.append(bagging_func(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size,num_trees=num_trees))

        
        print('svm')
        err_svm.append(support_vector(X_train.copy(),Y_train.copy(),X_test.copy(),Y_test.copy()))
        ################
    print('svm {} '.format(np.average(err_svm)))
    print('boost {} '.format(np.average(err_boost)))
    print('rand {} '.format(np.average(err_rand)))
    print('bag {} '.format(np.average(err_bag)))
    print('decision {} '.format(np.average(err_decision)))
    
    err_svm_arr.append(err_svm)
    err_boost_arr.append(err_boost)
    err_rand_arr.append(err_rand)
    err_decision_arr.append(err_decision)
    err_bag_arr.append(err_bag)
    
        


